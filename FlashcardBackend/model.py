from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load model
model_path = 'FlashcardBackend/fine_tuned_bert_sst2_2.pth'  # Adjust the path to your model file
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the fine-tuned model weights
# model.load_state_dict(torch.load(model_path, map_location=model.device))

# Function to predict the sentiment
def predict_sentiment(input_text):
    model.eval()
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()  # Get the predicted class
    # model.train()  # Uncomment if you plan to further train the model
    return prediction

# Test the function
input_string = "Though he is not a good writer, he is a brilliant cricketer"
predicted_class = predict_sentiment(input_string)
print(f"Predicted sentiment for the input is: {'Positive' if predicted_class == 1 else 'Negative'}")
