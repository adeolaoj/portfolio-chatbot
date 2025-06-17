# Portfolio Chatbot (Terminal Version)

## 💬 Description
This is a terminal-based version of the chatbot featured on my portfolio website. You can interact with it by asking about my skills, projects, fun facts, and more.

## 🛠️ Technical Details
- **Natural Language Processing**: User input is tokenized, stemmed (using NLTK's PorterStemmer), and converted into a bag-of-words vector.
- **Model Architecture**: A sequential neural network with:
  - Input layer
  - Hidden layer (ReLU activation)
  - Output layer (softmax for intent classification)
- **Frameworks Used**:
  - TensorFlow & Keras for model training and inference
  - NLTK for preprocessing
- **Model Persistence**: Trained model is saved in `.keras` format, with associated metadata stored in `.pkl` files.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow nltk

2. Run the chatbot:
   python chat.py

## Files Included
- intents.json: Contains the intent patterns and responses
- train_chatbot.py: Trains and saves the chatbot model
- chat.py: Runs the chatbot in the terminal
- pkl files: Store preprocessed training data (words, classes, etc.)
- chatbot_model.keras: Trained model file
