# peft-llama

Demo train a LoRA adapater for `meta-llama/Llama-3.2-1B-Instruct` model using `neuralwork/fashion-style-instruct` dataset using a single Nvidia L4 (24Gi) with PyTorch CUDA v12.6, Python v3.11, PyTorch v2.6 workbench in RHOAI.

Install deps.

```bash
pip install -r requirements.txt
```

Train.

```bash
export HF_TOKEN=hf_
python train.py
```

Training output.

```bash
 67%|█████████████████████████████████▎                | 222/333 [32:41<12:08,  6.56s/it]/opt/app-root/lib64/python3.11/site-packages/peft/utils/save_and_load.py:220: UserWarning: Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.
  warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
{'loss': 0.6233, 'grad_norm': 0.46154651045799255, 'learning_rate': 0.0002, 'num_tokens': 3726878.0, 'mean_token_accuracy': 0.8098986776251542, 'epoch': 2.07}
{'loss': 0.615, 'grad_norm': 0.4267504811286926, 'learning_rate': 0.0002, 'num_tokens': 3890718.0, 'mean_token_accuracy': 0.8111565560102463, 'epoch': 2.16}
{'loss': 0.6167, 'grad_norm': 0.4349284768104553, 'learning_rate': 0.0002, 'num_tokens': 4054558.0, 'mean_token_accuracy': 0.8117549926042557, 'epoch': 2.25}
{'loss': 0.6181, 'grad_norm': 0.4331114590167999, 'learning_rate': 0.0002, 'num_tokens': 4216999.0, 'mean_token_accuracy': 0.8105039924383164, 'epoch': 2.34}
{'loss': 0.6164, 'grad_norm': 0.46278610825538635, 'learning_rate': 0.0002, 'num_tokens': 4379511.0, 'mean_token_accuracy': 0.8107595384120941, 'epoch': 2.43}
{'loss': 0.6224, 'grad_norm': 0.4391016364097595, 'learning_rate': 0.0002, 'num_tokens': 4543351.0, 'mean_token_accuracy': 0.8091353058815003, 'epoch': 2.52}
{'loss': 0.6232, 'grad_norm': 0.42695218324661255, 'learning_rate': 0.0002, 'num_tokens': 4707191.0, 'mean_token_accuracy': 0.8089398980140686, 'epoch': 2.62}
{'loss': 0.6196, 'grad_norm': 0.4080263078212738, 'learning_rate': 0.0002, 'num_tokens': 4871031.0, 'mean_token_accuracy': 0.8100940257310867, 'epoch': 2.71}
{'loss': 0.6265, 'grad_norm': 0.4328615069389343, 'learning_rate': 0.0002, 'num_tokens': 5034871.0, 'mean_token_accuracy': 0.8079262197017669, 'epoch': 2.8}
{'loss': 0.6201, 'grad_norm': 0.41959264874458313, 'learning_rate': 0.0002, 'num_tokens': 5197927.0, 'mean_token_accuracy': 0.8088323682546615, 'epoch': 2.89}
{'loss': 0.6276, 'grad_norm': 0.4177376329898834, 'learning_rate': 0.0002, 'num_tokens': 5361767.0, 'mean_token_accuracy': 0.8082559704780579, 'epoch': 2.98}
100%|██████████████████████████████████████████████████| 333/333 [49:09<00:00,  6.55s/it]/opt/app-root/lib64/python3.11/site-packages/peft/utils/save_and_load.py:220: UserWarning: Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.
  warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
{'train_runtime': 2951.1883, 'train_samples_per_second': 0.896, 'train_steps_per_second': 0.113, 'train_loss': 0.7175508354519222, 'num_tokens': 5396583.0, 'mean_token_accuracy': 0.8139472246170044, 'epoch': 3.0}
100%|██████████████████████████████████████████████████| 333/333 [49:11<00:00,  8.86s/it]
/opt/app-root/lib64/python3.11/site-packages/peft/utils/save_and_load.py:220: UserWarning: Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.
  warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
(app-root) sh-5.1
```

LoRA adapter:

```bash
(app-root) sh-5.1$ ls -la Llama-3.2-1B-Instruct-style/
total 1147340
drwxr-sr-x.  6 1000910000 1000910000       4096 May 25 01:36 .
drwxrwsrwx. 10 root       1000910000        217 May 25 02:05 ..
-rw-r--r--.  1 1000910000 1000910000        880 May 25 01:36 adapter_config.json
-rw-r--r--.  1 1000910000 1000910000 1157560600 May 25 01:36 adapter_model.safetensors
-rw-r--r--.  1 1000910000 1000910000       3827 May 25 01:36 chat_template.jinja
drwxr-sr-x.  2 1000910000 1000910000       4096 May 25 00:23 checkpoint-111
drwxr-sr-x.  2 1000910000 1000910000       4096 May 25 00:40 checkpoint-222
drwxr-sr-x.  2 1000910000 1000910000       4096 May 25 00:56 checkpoint-333
-rw-r--r--.  1 1000910000 1000910000        184 May 25 01:36 generation_config.json
-rw-r--r--.  1 1000910000 1000910000       5174 May 25 01:36 README.md
drwxr-sr-x.  3 1000910000 1000910000         34 May 25 00:07 runs
-rw-r--r--.  1 1000910000 1000910000        296 May 25 01:36 special_tokens_map.json
-rw-r--r--.  1 1000910000 1000910000      50521 May 25 01:36 tokenizer_config.json
-rw-r--r--.  1 1000910000 1000910000   17209920 May 25 01:36 tokenizer.json
-rw-r--r--.  1 1000910000 1000910000       6097 May 25 00:56 training_args.bin
```

Push to HF

```bash
python save_hf.py
```

Test using a random sample from dataset - output:

```bash
(app-root) sh-5.1$ python chat.py 
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
/opt/app-root/lib64/python3.11/site-packages/peft/tuners/tuners_utils.py:550: UserWarning: Model with `tie_word_embeddings=True` and the tied_target_modules=['lm_head'] are part of the adapter. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. See for example https://github.com/huggingface/peft/issues/2018.
  warnings.warn(
Dataset({
    features: ['input', 'completion', 'context'],
    num_rows: 3193
})
You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
        ### Input:
        I'm a plus-size, short man with a preference for bold, statement pieces. I like brightly colored shirts and jackets that reflect my personality.

        ### Context:
        I'm going to a night club.

        ### Response:

    
Instruction: 
I'm a plus-size, short man with a preference for bold, statement pieces. I like brightly colored shirts and jackets that reflect my personality.

Context: 
I'm going to a night club.

Ground truth: 
Outfit 1:
- Top: A bold, bright patterned button-down shirt in a vibrant color like red or royal blue. Opt for a slimming fit that accentuates your shape.
- Bottom: Pair the shirt with dark wash jeans that have a slim leg. This will balance out the vibrant top and create a sleek look.
- Shoes: Go for a pair of black leather Chelsea boots. They are stylish and versatile, perfect for a night out.
- Accessories: Add a statement belt with metallic details and a chunky silver watch to add some flair to the outfit.

Outfit 2:
- Top: Choose a stylish and eye-catching bomber jacket in a bold color like mustard yellow or emerald green. The bomber jacket should be slightly oversized to create a trendy look.
- Bottom: Opt for black tailored trousers that have a slim, tapered fit. They will create a streamlined silhouette and complement the colorful jacket.
- Shoes: Wear black patent leather dress shoes with a slightly pointed toe to add a touch of sophistication to the outfit.
- Accessories: Add a black leather bracelet with metallic accents and a pair of statement sunglasses to complete the look.

Outfit 3:
- Top: Pick a bright, graphic print t-shirt that highlights your personality. Look for unique patterns or designs that make a statement.
- Bottom: Pair the t-shirt with dark wash slim-fit jeans. The contrast with the vibrant top will provide a balanced and flattering look.
- Shoes: Opt for white low-top sneakers with colorful accents. They will add a trendy and playful touch to the ensemble.
- Accessories: Add a black leather backpack with bold hardware details and choose a colorful beaded bracelet as a subtle accessory.

Outfit 4:
- Top: Wear a brightly colored, slim-fit dress shirt in a vibrant shade like fuchsia or cobalt blue. Look for interesting details like contrasting buttons or a textured fabric.
- Bottom: Pair the dress shirt with tailored charcoal gray dress pants. The combination of colors will create a modern and sophisticated outfit.
- Shoes: Choose black leather loafers with a bit of shine. They are versatile and will add a polished touch to your look.
- Accessories: Add a sleek black leather belt with a metallic buckle and a pocket square in a complementary color to complete the outfit.

Outfit 5:
- Top: Opt for a bold, bright blazer in a standout color like electric purple or fiery orange. Look for a tailored fit to enhance your shape.
- Bottom: Wear black straight-leg trousers that have a slightly relaxed fit. The contrast with the vibrant blazer will draw attention to your upper body.
- Shoes: Go for black suede oxford shoes. Their elegant yet trendy design will balance out the boldness of the blazer.
- Accessories: Add a silver lapel pin that matches your personality, and a slim black leather wallet with metallic accents for a sleek finishing touch.

Generated output: 
 Outfit 1:
- Top: A fitted, solid-colored button-down shirt in a bold, vibrant color such as royal blue or fuchsia.
- Bottom: Dark wash, high-waisted jeans with a slim fit to elongate your legs.
- Shoe: Black, chunky platform boots for a stylish and edgy look.
- Accessories: A silver chain necklace with a statement pendant and a black leather belt to cinch the shirt at the waist.

Outfit 2:
- Top: A patterned, oversized graphic t-shirt with a bold print or logo.
- Bottom: Black, tailored trousers with a cropped length to show off your height.
- Shoe: White, low-top sneakers for a modern and casual vibe.
- Accessories: A black beanie hat and a silver wristwatch to add a touch of edginess.

Outfit 3:
- Top: A long-sleeved, solid-colored Henley shirt with a subtle sheen.
- Bottom: Olive green, high-waisted cargo pants for a relaxed and stylish look.
- Shoe: Brown, ankle boots with a chunky heel for added height and confidence.
- Accessories: A brown leather belt with a silver buckle and a pair of aviator sunglasses.

Outfit 4:
- Top: A fitted, solid-colored polo shirt in a bold color such as red or purple.
- Bottom: Gray, slim-fit chinos with a tailored appearance.
- Shoe: Tan, suede loafers for a sophisticated and laid-back look.
- Accessories: A silver chain bracelet and a black leather watch to complement the polo shirt.

Outfit 5:
- Top: A long-sleeved, patterned button-down shirt with a bold print such as snakes or leopard.
- Bottom: Black, ankle-length leggings for a sleek and edgy appearance.
- Shoe: White, low-top black leather sneakers for a modern and fashion-forward look.
- Accessories: A black leather belt with a silver buckle and a silver pendant necklace to add a touch of elegance.

Remember, these outfit combinations are just suggestions to help you create a stylish and confident look. Play with different combinations and tailor them to your personal style and preferences.
    
```
