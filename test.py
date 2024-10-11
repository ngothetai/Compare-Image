from MIC.model.instructblip import InstructBlipConfig,InstructBlipForConditionalGeneration,InstructBlipProcessor
from PIL import Image
import torch
model_type="instructblip"
model_ckpt="BleachNick/MMICL-Instructblip-T5-xl"
processor_ckpt = "Salesforce/instructblip-flan-t5-xl"
config = InstructBlipConfig.from_pretrained(model_ckpt)

if 'instructblip' in model_type:
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_ckpt,
        config=config).to('cuda:0',dtype=torch.bfloat16)

image_palceholder="图"
sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
processor = InstructBlipProcessor.from_pretrained(
    processor_ckpt
)
sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
    model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
replace_token="".join(32*[image_palceholder])


image = Image.open ("data/val/v1_1_before.png")
image1 = Image.open ("data/val/v1_1_after.png")
images = [image,image1]

prompt = [f'Use the image before: <image0>{replace_token},image after: <image1>{replace_token}'
          f'The image before is the on the left and the image after is on the right.'
          f'Given the image before and image after as visual aid to answer the following question:'
          f'The hem of the vestment in image 1 is softer than in image 2. Which is image 2: After or Before?']
prompt = " ".join(prompt)

inputs = processor(images=images, text=prompt, return_tensors="pt")

inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

inputs = inputs.to('cuda:0')
outputs = model.generate(
        pixel_values = inputs['pixel_values'],
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        img_mask = inputs['img_mask'],
        do_sample=False,
        max_length=50,
        min_length=1,
        set_min_padding_size =False,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
