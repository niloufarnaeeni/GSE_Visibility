
# ه®‰è£…çژ¯ه¢ƒ

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#ن¸؛ن؛†éپ؟ه…چè‡ھهٹ¨ه®‰è£…çڑ„torchن¸ژوœ¬هœ°çڑ„cudaن¸چه…¼ه®¹ï¼Œه»؛è®®è؟›è،Œن¸‹ن¸€و­¥ن¹‹ه‰چه…ˆو‰‹هٹ¨ه®‰è£…وœ¬هœ°cudaç‰ˆوœ¬ه…¼ه®¹çڑ„torchم€‚
pip install -r requirements.txt 
```
    
               

# ه¾®è°ƒو¨،ه‍‹

هœ¨ه®‰è£…ه¥½ن¾‌èµ–هگژï¼Œوˆ‘ن»¬é€ڑè؟‡ه…·ن½“çڑ„ç¤؛ن¾‹و‌¥ه±•ç¤؛ه¦‚ن½•هˆ©ç”¨وˆ‘ن»¬è‡ھه·±çڑ„و•°وچ®و‌¥ه¾®è°ƒه¼€و؛گçڑ„وژ’ه؛ڈو¨،ه‍‹ (BAAI/bge-reranker-v2-m3)ï¼Œوˆ–è€…ن»ژ BERT ç±»و¨،ه‍‹ (hfl/chinese-roberta-wwm-ext) ن»¥هڈٹ LLM ç±»و¨،ه‍‹ (Qwen/Qwen2.5-1.5B) ن»ژé›¶ه¼€ه§‹è®­ç»ƒوژ’ه؛ڈو¨،ه‍‹م€‚ن¸ژو­¤هگŒو—¶ï¼Œوˆ‘ن»¬ن¹ںو”¯وŒپه°† LLM ç±»و¨،ه‍‹çڑ„وژ’ه؛ڈèƒ½هٹ›è’¸é¦ڈهˆ°è¾ƒه°ڈçڑ„ BERT و¨،ه‍‹ن¸­هژ»م€‚


## و•°وچ®هٹ è½½

وˆ‘ن»¬وڈگن¾›ن¸¤ç§چو•°وچ®é›†هٹ è½½و–¹ه¼ڈï¼Œç”¨ن؛ژو”¯وŒپن¸چهگŒç±»ه‍‹çڑ„وچںه¤±ه‡½و•°ï¼ڑ

### هچ•ç‚¹و•°وچ®هٹ è½½

هچ•ç‚¹و•°وچ®çڑ„و ¼ه¼ڈï¼ڑéƒ½وک¯ن¸€ن¸ھqueryï¼Œن¸€ن¸ھdocï¼Œه’Œه…¶ه¯¹ه؛”çڑ„labelم€‚ç¤؛ن¾‹è§پ [pointwise_reranker_train_data.jsonl](../../../example_data/pointwise_reranker_train_data.jsonl)
```
{"query": str, "content": str, "label": int|float}
```
- `content` وک¯ query و‰€ه¯¹ه؛”çڑ„و–‡و،£ه®‍é™…ه†…ه®¹م€‚
- `label` وک¯و¨،ه‍‹ه¾®è°ƒçڑ„ç›‘ç‌£ن؟،هڈ·ï¼Œوœ‰ن¸¤ç§چç±»ه‍‹ï¼ڑ
  - è؟‍ç»­ه‍‹ï¼ڑ0-1 ن¹‹é—´çڑ„è؟‍ç»­ه€¼هˆ†و•°
  - ç¦»و•£ه‍‹ï¼ڑه¤ڑç؛§ç›¸ه…³و€§و ‡ç­¾ï¼ˆ0/1/2/...ï¼‰ï¼Œو•°وچ®هٹ è½½و¨،ه‌—ن¼ڑه°†ه…¶ه‌‡هŒ€و”¾ç¼©وˆگ 0-1 هŒ؛é—´çڑ„è؟‍ç»­ه€¼م€‚

> ه½“ç›¸ه…³و€§وک¯ه¤ڑç؛§و ‡ç­¾و—¶ï¼Œé€ڑè؟‡è®¾ه®ڑ `max_label` ه’Œ `min_label` ï¼Œو•°وچ®é›†ه†…éƒ¨ن¼ڑè‡ھهٹ¨ه°†ه¤ڑç؛§و ‡ç­¾ه‌‡هŒ€و”¾ç¼©هˆ° 0-1 هˆ†و•°هŒ؛é—´ن¸­م€‚
> ن¾‹ه¦‚و•°وچ®é›†ن¸­ه­کهœ¨ن¸‰ç؛§و ‡ç­¾ï¼ˆ0ï¼Œ1ï¼Œ2ï¼‰ï¼Œç»ڈè؟‡و”¾ç¼©هگژï¼Œه¾—هˆ°ï¼ڑ{ label 0: 0ï¼Œlabel 1: 0.5ï¼Œlabel 2: 1 }م€‚

و”¯وŒپن¸¤ç§چوچںه¤±ه‡½و•°ï¼ڑن؛Œهˆ†ç±»ن؛¤هڈ‰ç†µوچںه¤±ï¼ˆBCEï¼‰ï¼ˆن؛Œهˆ†ç±»هœ؛و™¯ه’Œsoft labelن¸‹çڑ„ن؛¤هڈ‰ç†µlossه‌‡هڈ¯ï¼‰ه’Œه‌‡و–¹è¯¯ه·®وچںه¤±ï¼ˆMSEï¼‰م€‚


### هˆ†ç»„و•°وچ®هٹ è½½

هˆ†ç»„و•°وچ®çڑ„و ¼ه¼ڈï¼ڑéƒ½وک¯ن¸€ن¸ھqueryï¼Œن¸€ç»„docن»¥هڈٹه¯¹ه؛”çڑ„labelم€‚ç¤؛ن¾‹è§پ [grouped_reranker_train_data_listwise_label.jsonl](../../../example_data/grouped_reranker_train_data_listwise_label.jsonl)

```
{"query": str, "hits": [{"content": str, "label": int|float}, ...]}
```
- `hits` ن¸؛ query ه¯¹ه؛”çڑ„و‰€وœ‰و–‡و،£و ·وœ¬ï¼Œcontent وک¯و–‡و،£çڑ„ه®‍é™…ه†…ه®¹م€‚
- `label` وک¯و¨،ه‍‹ه¾®è°ƒçڑ„ç›‘ç‌£ن؟،هڈ·ï¼Œهڈ¯ن»¥وœ‰ن¸¤ç§چç±»ه‍‹ï¼ڑ
  - è؟‍ç»­ه‍‹ï¼ڑ0-1 ن¹‹é—´çڑ„è؟‍ç»­ه€¼هˆ†و•°
  - ç¦»و•£ه‍‹ï¼ڑه¤ڑç؛§ç›¸ه…³و€§و ‡ç­¾ï¼ˆ0/1/2/...ï¼‰

**وˆگه¯¹وژ’هگچوچںه¤±ï¼ˆRankNet Lossï¼‰ï¼ڑ**

هœ¨è¯¥lossن¸‹ï¼Œrag-retrievalن¼ڑè‡ھهٹ¨و ¹وچ®queryه’Œdocه¯¹ه؛”çڑ„labelï¼Œè‡ھهٹ¨ç»„وˆگpairè®،ç®—lossï¼Œه¹¶ن½؟ç”¨ن¸¤è€…çڑ„ه·®ه€¼è؟›è،Œهٹ و‌ƒم€‚

```math
\mathcal{L}_\mathrm{RankNet}= \sum_{i=1}^M\sum_{j=1}^M \mathbb{1}_{r_{i} < r_{j} } \ |r_j-r_i|\ \log(1 + \exp(s_i-s_j))
```
  

## è®­ç»ƒ

BERT ç±»و¨،ه‍‹è®­ç»ƒ, fsdp(ddp)
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
> ./logs/training_bert.log &
```

LLM ç±»و¨،ه‍‹è®­ç»ƒ, deepspeedï¼ˆن»…é€‚ç”¨ن؛ژzero 1-2, zero 3 وڑ‚ن¸چé€‚é…چم€گن؟‌ه­کو¨،ه‍‹çڑ„و—¶ه€™وœ‰ bugم€‘ï¼‰
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
> ./logs/training_llm_deepspeed1.log &
```

## **هڈ‚و•°è§£é‡ٹ**

ه¤ڑهچ،è®­ç»ƒconfig_file:

- ه¯¹ن؛ژ BERT ç±»و¨،ه‍‹ï¼Œé»کè®¤ن½؟ç”¨fsdpو‌¥و”¯وŒپه¤ڑهچ،è®­ç»ƒو¨،ه‍‹ï¼Œن»¥ن¸‹وک¯é…چç½®و–‡ن»¶çڑ„ç¤؛ن¾‹م€‚
  - [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml)ï¼ڑ ه¦‚و‍œè¦پهœ¨ hfl/chinese-roberta-wwm-ext çڑ„هں؛ç،€ن¸ٹن»ژé›¶ه¼€ه§‹è®­ç»ƒçڑ„وژ’ه؛ڈï¼Œé‡‡ç”¨è¯¥é…چç½®و–‡ن»¶
  -  [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml)ï¼ڑ ه¦‚و‍œè¦پهœ¨ BAAI/bge-reranker-baseم€پmaidalun1020/bce-reranker-base_v1م€پBAAI/bge-reranker-v2-m3 çڑ„هں؛ç،€ن¸ٹè؟›è،Œه¾®è°ƒï¼Œé‡‡ç”¨è¯¥é…چç½®و–‡ن»¶ï¼Œه› ن¸؛ه…¶éƒ½وک¯هœ¨ه¤ڑè¯­è¨€çڑ„ XLMRoberta çڑ„هں؛ç،€ن¸ٹè®­ç»ƒè€Œو‌¥

- ه¯¹ن؛ژ LLM ç±»و¨،ه‍‹ï¼Œه»؛è®®ن½؟ç”¨ deepspeed و‌¥و”¯وŒپه¤ڑهچ،è®­ç»ƒو¨،ه‍‹ï¼Œç›®ه‰چهڈھو”¯وŒپ zero1 ه’Œ zero2 çڑ„è®­ç»ƒéک¶و®µï¼Œن»¥ن¸‹وک¯é…چç½®و–‡ن»¶çڑ„ç¤؛ن¾‹
  - [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
  - [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

- ه¤ڑهچ،è®­ç»ƒé…چç½®و–‡ن»¶ن؟®و”¹:
  - ن؟®و”¹ه‘½ن»¤ن¸­çڑ„ CUDA_VISIBLE_DEVICES="0" ن¸؛ن½ وƒ³è¦پè®¾ç½®çڑ„ه¤ڑهچ،
  - ن؟®و”¹ن¸ٹè؟°وڈگهˆ°çڑ„é…چç½®و–‡ن»¶çڑ„ num_processes ن¸؛ن½ وƒ³è¦پè·‘çڑ„هچ،çڑ„و•°é‡ڈ


و¨،ه‍‹و–¹é‌¢ï¼ڑ
- `model_name_or_path`ï¼ڑه¼€و؛گçڑ„rerankerو¨،ه‍‹çڑ„هگچç§°وˆ–ن¸‹è½½ن¸‹و‌¥çڑ„وœ¬هœ°وœچهٹ،ه™¨ن½چç½®م€‚ن¾‹ه¦‚ï¼ڑBAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1ï¼Œن¹ںهڈ¯ن»¥ن»ژé›¶ه¼€ه§‹è®­ç»ƒï¼Œن¾‹ه¦‚BERT: hfl/chinese-roberta-wwm-ext ه’ŒLLM: Qwen/Qwen2.5-1.5Bï¼‰
- `model_type`ï¼ڑه½“ه‰چو”¯وŒپ bert_encoderوˆ–llm_decoderç±»و¨،ه‍‹
- `max_len`ï¼ڑو•°وچ®و”¯وŒپçڑ„وœ€ه¤§è¾“ه…¥é•؟ه؛¦

و•°وچ®é›†و–¹é‌¢ï¼ڑ
- `train_dataset`ï¼ڑè®­ç»ƒو•°وچ®é›†ï¼Œه…·ن½“و ¼ه¼ڈè§پن¸ٹو–‡
- `val_dataset`ï¼ڑéھŒè¯پو•°وچ®é›†ï¼Œو ¼ه¼ڈهگŒè®­ç»ƒé›†(ه¦‚و‍œو²،وœ‰ï¼Œè®¾ç½®ن¸؛ None هچ³هڈ¯)
- `max_label`ï¼ڑهچ•ç‚¹و•°وچ®é›†ن¸­çڑ„وœ€ه¤§ labelï¼Œé»کè®¤ن¸؛ 1
- `min_label`ï¼ڑهچ•ç‚¹و•°وچ®é›†ن¸­çڑ„وœ€ه°ڈ labelï¼Œé»کè®¤ن¸؛ 0

è®­ç»ƒو–¹é‌¢ï¼ڑ
- `output_dir`ï¼ڑè®­ç»ƒè؟‡ç¨‹ن¸­ن؟‌ه­کçڑ„ checkpoint ه’Œوœ€ç»ˆو¨،ه‍‹çڑ„ç›®ه½•
- `loss_type`ï¼ڑé€‰و‹©وœ€ç»ˆè®؛و–‡ن¸­çڑ„وژ’ه؛ڈç›®و ‡ï¼ڑ`ranknet`ï¼ˆRankNetï¼‰م€پ`ear`ï¼ˆEARï¼‰م€پ`ear_sym`ï¼ˆEAR-Symï¼‰وˆ– `pairwise_reg`ï¼ˆPairwise Regï¼‰م€‚
- `epoch`ï¼ڑو¨،ه‍‹هœ¨è®­ç»ƒو•°وچ®é›†ن¸ٹè®­ç»ƒçڑ„è½®و•°
- `lr`ï¼ڑه­¦ن¹ çژ‡ï¼Œن¸€èˆ¬1e-5هˆ°5e-5ن¹‹é—´
- `batch_size`ï¼ڑو¯ڈن¸ھ batch ن¸­ query-doc pair ه¯¹çڑ„و•°é‡ڈ
- `seed`ï¼ڑè®¾ç½®ç»ںن¸€ç§چه­گï¼Œç”¨ن؛ژه®‍éھŒç»“و‍œçڑ„ه¤چçژ°
- `warmup_proportion`ï¼ڑه­¦ن¹ çژ‡é¢„çƒ­و­¥و•°هچ و¨،ه‍‹و›´و–°و€»و­¥و•°çڑ„و¯”ن¾‹ï¼Œه¦‚و‍œè®¾ç½®ن¸؛ 0ï¼Œé‚£ن¹ˆن¸چè؟›è،Œه­¦ن¹ çژ‡é¢„çƒ­ï¼Œç›´وژ¥ن»ژè®¾ç½®çڑ„ `lr` è؟›è،Œن½™ه¼¦è،°é€€
- `stable_proportion`ï¼ڑه­¦ن¹ çژ‡ç¨³ه®ڑن¸چهڈکçڑ„و­¥و•°هچ و¨،ه‍‹و›´و–°و€»و­¥و•°çڑ„و¯”ن¾‹ï¼Œé»کè®¤وک¯ 0
- `gradient_accumulation_steps`ï¼ڑو¢¯ه؛¦ç´¯ç§¯و­¥و•°ï¼Œو¨،ه‍‹ه®‍é™…çڑ„ batch_size ه¤§ه°ڈç­‰ن؛ژ `batch_size` * `gradient_accumulation_steps` * `num_of_GPUs`
- `mixed_precision`ï¼ڑوک¯هگ¦è؟›è،Œو··هگˆç²¾ه؛¦çڑ„è®­ç»ƒï¼Œن»¥é™چن½ژوک¾ه­کçڑ„éœ€و±‚م€‚و··هگˆç²¾ه؛¦è®­ç»ƒé€ڑè؟‡هœ¨è®،ç®—ن½؟ç”¨ن½ژç²¾ه؛¦ï¼Œو›´و–°هڈ‚و•°ç”¨é«کç²¾ه؛¦ï¼Œو‌¥ن¼کهŒ–وک¾ه­کهچ ç”¨م€‚ه¹¶ن¸” bf16ï¼ˆBrain Floating Point 16ï¼‰هڈ¯ن»¥وœ‰و•ˆé™چن½ژ loss scaling çڑ„ه¼‚ه¸¸وƒ…ه†µï¼Œن½†è¯¥ç±»ه‍‹ن»…è¢«éƒ¨هˆ†ç،¬ن»¶و”¯وŒپ
- `save_on_epoch_end`ï¼ڑوک¯هگ¦هœ¨و¯ڈن¸€ن¸ھ epoch ç»“و‌ںهگژéƒ½ن؟‌ه­کو¨،ه‍‹
- `num_max_checkpoints`ï¼ڑوژ§هˆ¶هچ•و¬،è®­ç»ƒن¸‹ن؟‌ه­کçڑ„وœ€ه¤ڑ checkpoints و•°ç›®
- `log_interval`ï¼ڑو¨،ه‍‹و¯ڈو›´و–° x و¬،هڈ‚و•°è®°ه½•ن¸€و¬، loss
- `log_with`ï¼ڑهڈ¯è§†هŒ–ه·¥ه…·ï¼Œن»ژ wandb ه’Œ tensorboard ن¸­é€‰و‹©

و¨،ه‍‹هڈ‚و•°ï¼ڑ
- `num_labels`ï¼ڑو¨،ه‍‹è¾“ه‡؛ logit çڑ„و•°ç›®ï¼Œهچ³ن¸؛و¨،ه‍‹هˆ†ç±»ç±»هˆ«çڑ„ن¸ھو•°ï¼Œن¸€èˆ¬é»کè®¤è®¾ç½®ن¸؛ 1
- ه¯¹ن؛ژ LLM ç”¨ن؛ژهˆ¤هˆ«ه¼ڈوژ’ه؛ڈو‰“هˆ†و—¶ï¼Œéœ€è¦پن؛؛ه·¥و‍„é€ è¾“ه…¥و ¼ه¼ڈï¼Œç”±و­¤ه¼•ه…¥ن¸‹هˆ—هڈ‚و•°
  - `query_format`, e.g. "query: {}"
  - `document_format`, e.g. "document: {}" 
  - `seq`ï¼ڑهˆ†éڑ” query ه’Œ document éƒ¨هˆ†, e.g. " "
  - `special_token`ï¼ڑé¢„ç¤؛ç‌€ document ه†…ه®¹çڑ„ç»“و‌ںï¼Œه¼•ه¯¼و¨،ه‍‹ه¼€ه§‹و‰“هˆ†ï¼Œçگ†è®؛ن¸ٹهڈ¯ن»¥وک¯ن»»ن½• token, e.g. "\</s>" 
  - و•´ن½“çڑ„و ¼ه¼ڈن¸؛ï¼ڑ"query: xxx document: xxx\</s>" 


# هٹ è½½و¨،ه‍‹è؟›è،Œé¢„وµ‹

ه¯¹ن؛ژن؟‌ه­کçڑ„و¨،ه‍‹ï¼Œن½ هڈ¯ن»¥ه¾ˆه®¹وک“هٹ è½½و¨،ه‍‹و‌¥è؟›è،Œé¢„وµ‹م€‚

Cross-Encoder و¨،ه‍‹ï¼ˆBERT-likeï¼‰
```python
ckpt_path = "./bge-reranker-m3-base"
reranker = CrossEncoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["وˆ‘ه–œو¬¢ن¸­ه›½", "وˆ‘ه–œو¬¢ن¸­ه›½"],
    ["وˆ‘ه–œو¬¢ç¾ژه›½", "وˆ‘ن¸€ç‚¹éƒ½ن¸چه–œو¬¢ç¾ژه›½"]
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```

LLM-Decoder و¨،ه‍‹ ï¼ˆهں؛ن؛ژ MLP è؟›è،Œو ‡é‡ڈوک ه°„ï¼‰

> ن¸؛ن؛†و»،è¶³ LLM ه¦‚ "Qwen/Qwen2.5-1.5B" ç”¨ن؛ژهˆ¤هˆ«ه¼ڈوژ’ه؛ڈçڑ„ç‰¹و®ٹوƒ…ه†µï¼Œè®¾è®،ن؛†ç›¸ه…³و ¼ه¼ڈï¼Œه®‍é™…و•ˆو‍œن¸؛ï¼ڑ"query: {xxx} document: {xxx}\</s>"ï¼Œه®‍éھŒوک¾ç¤؛ \</s> çڑ„ه¼•ه…¥ه¯¹ LLM وژ’ه؛ڈو€§èƒ½وڈگهچ‡è¾ƒه¤§ [و؛گن؛ژ https://arxiv.org/abs/2411.04539 section 4.3]م€‚

```python
ckpt_path = "./Qwen2-1.5B-Instruct"
reranker = LLMDecoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
    query_format="query: {}",
    document_format="document: {}",
    seq="\n",
    special_token="\nrelevance",
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["وˆ‘ه–œو¬¢ن¸­ه›½", "وˆ‘ه–œو¬¢ن¸­ه›½"],
    ["وˆ‘ه–œو¬¢ç¾ژه›½", "وˆ‘ن¸€ç‚¹éƒ½ن¸چه–œو¬¢ç¾ژه›½"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```
