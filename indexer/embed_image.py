# library import
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import sys
import os

# utility import
from logger.logger import logging
from exception.exception import ProjectError

class SigLIPEmbedder:

    """ class for embedding the vectordb images and prompts/queries with siglip/dense embedding """
    
    def __init__(self, model_name=os.getenv("SIGLIP_MODEL", "google/siglip-base-patch16-224")):
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f" ----- Loading SigLIP model on {self.device} ----- ")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logging.error(f"Failed to load SigLIP model: {e}")
            raise ProjectError(str(e), sys)

    def embed_image(self, image: Image.Image) -> list[float]:
        
        """Extracts SigLIP image embeddings."""
        
        try:

            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_output = self.model.get_image_features(**inputs)

                if hasattr(image_output, "pooler_output"):
                    features = image_output.pooler_output
                elif hasattr(image_output, "last_hidden_state"):
                    features = image_output.last_hidden_state.mean(dim=1)
                else:
                    features = image_output

                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features[0].cpu().numpy().tolist()
        
        except Exception as e:
            logging.error(f"Error embedding image: {e}")
            raise ProjectError(str(e), sys)

    def embed_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        
        """Extracts SigLIP text embeddings."""
        
        try:
            
            if isinstance(text, str):
                text = [text]
                
            with torch.no_grad():
            
                inputs = self.processor(text=text, padding="max_length", return_tensors="pt").to(self.device)
                text_output = self.model.get_text_features(**inputs)
        
                if hasattr(text_output, "pooler_output"):
                    text_features = text_output.pooler_output
                elif hasattr(text_output, "last_hidden_state"):
                    text_features = text_output.last_hidden_state.mean(dim=1)
                else:
                    text_features = text_output
                
                # normalize the embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                if len(text) == 1:
                    return text_features.squeeze(0).cpu().numpy().tolist()
                
                return text_features.cpu().numpy().tolist()
        
        except Exception as e:
            logging.error(f"Error embedding text: {e}")
            raise ProjectError(str(e), sys)

# singleton instance

embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SigLIPEmbedder()
    return embedder
