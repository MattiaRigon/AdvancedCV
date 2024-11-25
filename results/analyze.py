import json
import torch
from torchmetrics import Metric
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class SemanticFScore:

    def __init__(self):
        self.cache_dir = ".cache"

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
        self._model = AutoModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self._model.eval()
        self._model.to(self.device)
        self.encode = self.encode_bert

    def compute_metrics(self,predictions, targets):
        x_tgts = torch.stack([self.encode(t) for t in targets])  # [M, D]
        x_vals = torch.stack([self.encode(v) for v in predictions])  # [N, D]

        x_tgts = F.normalize(x_tgts, dim=1)
        x_vals = F.normalize(x_vals, dim=1)

        s = x_tgts @ x_vals.T  # [M, N]
        r = s.max(dim=1).values.mean()  # [M]
        p = s.max(dim=0).values.mean()  # [N]
        f = torch.nan_to_num(2 * r * p / (r + p))

        return f, r, p

    def encode_bert(self, sentence: str, **kwargs) -> torch.Tensor:
        """
        Encode the input sentence with the BERT model.

        Args:
            sentence (str): Input sentence.
        """
        tokens = self._tokenizer.encode_plus(
            sentence,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        embeddings = self._model(**tokens).last_hidden_state

        # mask out padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask

        # sum over all tokens
        summed = torch.sum(masked_embeddings, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        # normalise and remove batch dimension
        embeddings = summed / summed_mask
        embeddings = embeddings.squeeze(0)

        return embeddings
# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to extract predictions for each image path
def extract_predictions(json1, json2):
    combined_predictions = {}

    for item in json1:
        image_path = item["image_path"]
        predictions_1 = item.get("predictions", [])
        combined_predictions[image_path] = {"predictions_1": predictions_1, "target": item.get("ground_truth", []), "image_path": image_path}

    for item in json2:
        image_path = item["image_path"]
        predictions_2 = item.get("predictions", [])
        if image_path in combined_predictions:
            combined_predictions[image_path]["predictions_2"] = predictions_2
        else:
            combined_predictions[image_path] = {"predictions_2": predictions_2}

    return combined_predictions


# Main execution
def main():
    # Replace with your JSON file paths
    json_file_1 = "evalutation_ours_coco_00000.json"
    json_file_2 = "evalutation_nxtp_coco_00000.json"

    # Read the JSON files
    json_data_1 = read_json(json_file_1)
    json_data_2 = read_json(json_file_2)

    # Extract predictions
    combined_predictions = extract_predictions(json_data_1, json_data_2)
    # Initialize SemanticFScore class
    semantic_fscore = SemanticFScore()

    results = {}

    # Print combined predictions
    for image_path, predictions in combined_predictions.items():
        print(f"Image Path: {image_path}")
        print(f"Predictions from JSON 1: {predictions.get('predictions_1', [])}")
        print(f"Predictions from JSON 2: {predictions.get('predictions_2', [])}")

        # Compute metrics
        f1, r1, p1 = semantic_fscore.compute_metrics(predictions.get('predictions_1', []), predictions.get('target', []))
        f2, r2, p2 = semantic_fscore.compute_metrics(predictions.get('predictions_2', []), predictions.get('target', []))

        results[image_path] = {
                                "f1_difference": abs(f1.item()-f2.item()) ,
                                "f1_1": f1.item(),
                                "recall_1": r1.item(),
                                "precision_1": p1.item(),
                                "f1_2": f2.item(),
                                "recall_2": r2.item(),
                                "precision_2": p2.item(),
                                "predictions_1": predictions.get('predictions_1', []),
                                "predictions_2": predictions.get('predictions_2', []),
                                "target": predictions.get('target', [])
                            }
        
    


        print("-")

    # Sort results by f1_difference
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]['f1_difference']))

    # Save results to a JSON file
    output_file = "sorted_results.json"
    with open(output_file, 'w') as file:
        json.dump(sorted_results, file, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
