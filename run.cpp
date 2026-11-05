#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
    #include <io.h>
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iostream>
#include <chrono>

struct Config {
    // number of features for each token embedding
    int dim;
    // inner dimension for feed-forward network
    // which is like an mlp
    int hidden_dim;
    // number of transformer blocks
    int n_layers;
    // number of attention blocks/"heads"
    // per transformer block
    int n_heads;
    // number of key/value blocks/"heads"
    // usually one per attention head
    // but multiquery so multiple queries share one kv head
    int n_kv_heads;
    // number of unique tokens that can be put out
    int vocab_size;
    // max sequence length
    int seq_len; 
};

//non-owning pointers into the mapped transformer weights
struct TransformerWeights {
    // token ID -> embedding vector contiguous mapping
    // (vocab_size,dim)
    float* token_embedding_table;
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm before attention, per layer
    float* rms_ffn_weight; // (layer, dim) rmsnorm before ffn, per layer
    // weights for matmuls
    // head_dim = dim / n_heads
    float* wq; // (layer, dim, dim) embedding => query projection for all heads combined
    float* wk; // (layer, dim or n_kv_heads*head_dim, dim) embedding => key projection for all heads combined
    float* wv; // (layer, dim or n_kv_heads*head_dim, dim) embedding => value projection for all heads combined
    float* wo; // (layer, dim, dim) attention result from all heads concated => embedding sized output 
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim) the "gated" projection
    float* w2; // (layer, dim, hidden_dim) the "down" projection
    float* w3; // (layer, hidden_dim, dim) the "up" projection
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis(cosine and sine) for RoPE relatively positional embeddings (rotational)
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls; //(vocab_size, dim);
};

struct RunContext {

    std::vector<float> x; // current token's main representation (dim,)
    std::vector<float> xb; // current token's residual branch which goes through attn (dim,)
    std::vector<float> xb2; // an additional buffer just for convenience (dim,)

    std::vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,) gated branch
    std::vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,) other branch

    // Q,K,V for the whole attention block, key and value being (dim,) and not (n_kv_heads * head_dim,)
    // indicates standard MHA
    std::vector<float> q; // query (dim,)
    std::vector<float> k; // key (dim,)
    std::vector<float> v; // value (dim,)

    std::vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    std::vector<float> logits; // output logits (vocab_size,)
    // kv cache
    std::vector<float> key_cache;   // (layer, seq_len, dim)
    std::vector<float> value_cache; // (layer, seq_len, dim)
};

void allocateRunContext(RunContext* ctx, Config* cfg) {

    ctx->x.resize(cfg->dim);
    ctx->xb.resize(cfg->dim);
    ctx->xb2.resize(cfg->dim);

    ctx->hb.resize(cfg->hidden_dim);
    ctx->hb2.resize(cfg->hidden_dim);

    ctx->q.resize(cfg->dim);
    ctx->k.resize(cfg->dim);
    ctx->v.resize(cfg->dim);

    ctx->att.resize(cfg->n_heads * cfg->seq_len);
    ctx->logits.resize(cfg->vocab_size);

    ctx->key_cache.resize(cfg->n_layers * cfg->seq_len * cfg->dim);
    ctx->value_cache.resize(cfg->n_layers * cfg->seq_len * cfg->dim);
}

struct WeightReader {
    float* ptr;

    float* take(size_t count) {
        float* start = ptr;
        ptr+=count;
        return start;
    }
};

//not allocating any memory or copying, creating pointers to the offsets in weights_ptr
/*
f / ptr
│
├─ token_embedding_table: vocab_size × dim floats
├─ rms_att_weight:       n_layers × dim floats
├─ wq:                   n_layers × dim × dim floats
├─ wk:                   n_layers × dim × dim floats
├─ wv:                   n_layers × dim × dim floats
├─ wo:                   n_layers × dim × dim floats
├─ rms_ffn_weight:       n_layers × dim floats
├─ w1:                   n_layers × dim × hidden_dim floats
├─ w2:                   n_layers × hidden_dim × dim floats
├─ w3:                   n_layers × dim × hidden_dim floats
├─ rms_final_weight:     dim floats
├─ freq_cis_real:        seq_len × (head_size / 2) floats
├─ freq_cis_imag:        seq_len × (head_size / 2) floats
└─ wcls:                 either shared embedding weights or remaining floats
*/
void initializeCheckpointWeights(TransformerWeights* weights, Config* config, float* weights_ptr, int shared_weights) {
    WeightReader wr{weights_ptr};

    weights->token_embedding_table = wr.take(config->vocab_size * config->dim);
    weights->rms_att_weight = wr.take(config->n_layers * config->dim);

    weights->wq = wr.take(config->n_layers * config->dim * config->dim);
    weights->wk = wr.take(config->n_layers * config->dim * config->dim);
    weights->wv = wr.take(config->n_layers * config->dim * config->dim);
    weights->wo = wr.take(config->n_layers * config->dim * config->dim);

    weights->rms_ffn_weight = wr.take(config->n_layers * config->dim);

    weights->w1 = wr.take(config->n_layers * config->dim * config->hidden_dim);
    weights->w2 = wr.take(config->n_layers * config->dim * config->hidden_dim);
    weights->w3 = wr.take(config->n_layers * config->dim * config->hidden_dim);

    weights->rms_final_weight = wr.take(config->dim);

    int head_size = config->dim / config->n_heads;
    weights->freq_cis_real = wr.take(config->seq_len * (head_size) / 2);
    weights->freq_cis_imag = wr.take(config->seq_len * (head_size) / 2);
    
    weights->wcls = shared_weights ? weights->token_embedding_table : wr.take(config->vocab_size * config->dim);
    return;

}

void bpe_encode(char* prompt, std::vector<std::string>& vocab, std::vector<float>& vocab_scores,
                int vocab_size, unsigned int max_token_length, std::vector<int>& prompt_tokens) {
    // if each word was a token, we would search the vocab for the word, and push the corresponding id into prompt_tokens
    // but this is byte pair encode, so we break down the prompt into bytes, then greedily merge according to the vocab_scores
    // when we can't merge anymore, we check the tokens and push the corresponding ids into prompt_tokens
    
    //split the prompt into byte tokens first
    std::string prmpt(prompt);
    std::vector<std::string> tokens;
    for(auto& ch : prmpt) {
        std::string byte{ch};
        tokens.push_back(byte);
    }

    for(auto& token : tokens) {
        auto it = std::find(vocab.begin(), vocab.end(), token);
    }
    do
    {
        // naive way
        // for each pair, check if present in vocab
        // if yes, check corresponding score
        // keep track of highest score, and the position of this score occurring
        // merge the highest priority
        // repeat
        // break if no further merges are possible
        // scores are negative
        float max_score = -40000.0f;
        int max_score_pos = -1;
        for(int i = 0; i < tokens.size() - 1; i++) {
            std::string pair = tokens[i] + tokens[i+1];
            // this is expensive, replace with a hashmap lookup
            auto it = std::find(vocab.begin(), vocab.end(), pair);
            if(it == vocab.end())
                continue;
            auto index = it - vocab.begin();
            if(vocab_scores[index] > max_score){
                max_score = vocab_scores[index];
                max_score_pos = i;
            }
        }
        if(max_score_pos == -1){
            break;
        }
        tokens[max_score_pos] = tokens[max_score_pos] + tokens[max_score_pos+1];
        tokens.erase(tokens.begin() + max_score_pos + 1);
    } while (1);
    //tokens vector should have the final strings. get corresponding ids in prompt_tokens
    for(auto& token : tokens) {
        auto it = std::find(vocab.begin(), vocab.end(), token);
        prompt_tokens.push_back(it - vocab.begin());
    }
    return;
}

void rmsnorm(std::vector<float>& xb, std::vector<float>& x, float* rms_attn_l) {

    // calculate root mean square 
    // RMS(x) = sqrt((x_1^2 + x_2^2 ... + x_dim^2) / n) 
    float rms = 0.0f;
    for(auto a : x){
        rms += a*a;
    }
    auto dim = x.size();
    // rms = sqrt(rms / (1.0f * dim));

    // // normalize x with RMS, scale with per-dim rms weight gamma
    // // RMSNorm(x_i) = gamma_i * x_i / RMS(x)
    // for(int i = 0; i < dim; i++) {
    //     xb[i] = *(rms_attn_l + i) * x[i];
    //     xb[i] /= rms;
    // }

    // corrections to above logic
    // repeated division -> bad
    // if RMS = 0, /=rms  will be undefined, add epsilon

    rms /= dim;
    rms += 1e-5f;
    auto inv_rms = 1.0f / std::sqrt(rms);

    for(int i = 0; i < dim; i++) {
        xb[i] = *(rms_attn_l + i) * x[i] * inv_rms;
    }
    return;
}

void transformer(int token, int pos, Config* config, RunContext* ctx, TransformerWeights* weights) {

    // token to vector embedding, store in x
    // assuming row major
    int dim = config->dim;
    auto embed = weights->token_embedding_table;
    // ctx->x = std::vector<float>(embed + token * dim,
    //      embed + token * config->dim + dim);
    std::copy_n(embed + token * dim, dim, ctx->x.begin());

    int head_size = dim / config->n_heads;

    // rotary positional embeddings values for pos
    // rotation looks like cos(theta) + i.sin(theta), real stores cosine terms, imag stores sine terms
    // RoPE discussed further below
    auto freq_cis_real_row = weights->freq_cis_real + pos * head_size / 2;
    auto freq_cis_imag_row = weights->freq_cis_imag + pos * head_size / 2;

    // forwarding loop for all layers
    // subsequent layer needs output x of previous layer
    for (int l = 0; l < config->n_layers; l++) {

        //attention rmsnorm
        rmsnorm(ctx->xb, ctx->x, weights->rms_att_weight + l * dim);
    }
    return;
}

void main(int argc, char *argv[]) {

    char* checkpoint{nullptr};
    float temperature{0.9f}; // controls sampling randomness
    int steps{256}; //number of iterations for generating tokens
    char* prompt{nullptr};

    //read from args
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps] [prompt]\n", argv[0]);
        return;
    }
    if (argc >= 2){
        checkpoint = argv[1];
    }
    if (argc >= 3){
        temperature = atof(argv[2]);
    }
    if (argc >= 4){
        steps = atoi(argv[3]);
    }
    if (argc >= 5){
        prompt = argv[4];
    }
    
    //read in the pytorch model weights and bias

    Config config;
    TransformerWeights weights;
    float* data{nullptr};
    long file_size;
    int fd = 0;
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) { printf("Couldn't open file %s\n", checkpoint); return; }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return; }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        fd = _open(checkpoint, O_RDONLY); // open in read only mode
        if (fd == -1) { printf("open failed!\n"); return; }
        data = static_cast<float*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) { printf("mmap failed!\n"); return; }
        float* weights_ptr = data + sizeof(Config)/sizeof(float);
        initializeCheckpointWeights(&weights, &config, weights_ptr, shared_weights);
    }
    
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }
    
    //load tokenizer.bin
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    //max length of a single token
    unsigned int max_token_length;
    {
        FILE *file = fopen("tokenizer.bin", "rb");
        if(!file) {
            printf("couldn't load tokenizer.bin\n");
            return;
        }
        if(fread(&max_token_length, sizeof(int), 1, file)!= 1) {
            printf("failed reading\n");
            return;
        }
        int len;
        // vocab info is stored as
        // vocab_score[i]|length[i]|vocab[i]
        for(int i = 0; i < config.vocab_size; i++) {
            float score;
            int length;
            std::string token;
            if (fread(&score,sizeof(float), 1, file) != 1) {printf("read failed\n"); return;}
            vocab_scores.push_back(score);
            if (fread(&length,sizeof(int),1,file)!= 1) {printf("read failed\n"); return;}
            token.resize(static_cast<size_t>(length));
            if (fread(token.data(),1,token.size(),file)!= token.size()) {printf("read failed\n"); return;}
            vocab.push_back(std::move(token));
        }
        fclose(file);            
    }

    // Execution context/state init
    RunContext ctx;
    // Allocate memory for the context
    allocateRunContext(&ctx, &config);


    printf("%d \n", vocab.size());
    printf("bpe encoding\n");

    // process the prompt if available
    std::vector<int> prompt_tokens;
    // int cnt = 0;
    // for(auto& word: vocab) {
    //     std::cout << "vocab :" << cnt << word << std::endl;
    //     if(cnt == 20)
    //         break;
    //     cnt++;
    // }
    if (prompt != NULL) {
        //prompt_tokens.resize(config.seq_len);
        const auto start = std::chrono::steady_clock::now();
        bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens);
        const auto end = std::chrono::steady_clock::now();
        std::cout << std::endl << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }
    //print the encoded tokens
    printf("%d tokens:\n", prompt_tokens.size());
    for (int i = 0; i < prompt_tokens.size(); i++) {
        printf("%d ",prompt_tokens[i]);
    }
    printf("\n");

    //main autoregressive loop
    auto start = std::chrono::steady_clock::now();
    int next; //stores next token in sequence
    int token = 1; //1 is the token id for BOS (beginning of sentence), llama2 sentencepiece tokenizer
    int pos = 0; // position in sequence (before processing the prompt)
    printf("<s>\n");
    while (pos < steps) {

        // forward the transformer to get logits for next token
        transformer(token, pos, &config, &ctx, &weights);

    }

    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) _close(fd);
    return;

        
}