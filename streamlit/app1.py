import streamlit as st 
import PIL
import torch 
from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.DEFAULT
transforms = weights.transforms()
model = torch.load('catdog.pt', map_location=torch.device('cpu'))

st.header("Cat or Dog???")
st.image('catdog.png')

st.text('''Below, upload a picture of your animal and we will predict if this
is a cat or a dog.''')

ims = st.file_uploader(label = 'picture of cat or dog', accept_multiple_files=True)
pred_dict = {0: 'Its a cat!', 1: 'It is a dog!!!'}
preds = []
if ims:
    for im in ims:
        
        imfile = PIL.Image.open(im)
    
        ready = transforms(imfile)
        pred = model(ready.unsqueeze(0)).argmax()
        preds.append(pred_dict[int(pred)])
    cols = st.columns(len(ims))
    for i, col in enumerate(cols):
        with col:
            st.image(ims[i], caption = pred_dict[int(pred)])
    