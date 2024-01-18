import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

class CustomBERTModel(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomBERTModel, self).__init__()
        # Load pre-trained BERT model for sequence classification
        self.bert = BertModel.from_pretrained('bert-base-uncased', num_labels=num_labels)

        # Define custom layers
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.linear2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get the output from BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the pooled output
        pooled_output = outputs[1]
        # Pass through custom layers
        linear1_output = self.linear1(pooled_output)
        linear2_output = self.linear2(linear1_output)

        return linear2_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the fine-tuned model
model_path = 'FlashcardBackend/fine_tuned_bert_sst2_2.pth'
model = torch.load(model_path, map_location=device)


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
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()  # Get the predicted class
    return prediction

def lambda_handler(event, context):
    # Parse the input data
    try:
        data = json.loads(event['body'])
        front_text = data['frontText']
        back_text = data['backText']

        # Predict sentiment
        front_text_pred = 'positive' if predict_sentiment(front_text) == 1 else 'negative'
        back_text_pred = 'positive' if predict_sentiment(back_text) == 1 else 'negative'

        # Return the response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'frontText': front_text,
                # 'frontTextPred': front_text_pred,
                'backText': front_text_pred
                # 'backTextPred': back_text_pred
            })
        }
        return response

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
# print(predict_sentiment("cool man"))