from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import simmpst
from simmpst.tokenization import MultilingualPartialSyllableTokenization
import random

def prepare_text(text,tokenizer):
    """
    Prepares the text for model input by tokenizing and padding it.
    
    Parameters:
    - text (str): The input text to be prepared.
    - tokenizer: The tokenizer object to tokenize the text.
    - maxlen (int): The maximum length to pad/truncate the sequences.
    
    Returns:
    - np.array: The padded tokenized text.
    - str: Error message if the text contains non-Burmese characters.
    """
    try:
        #check burmese or not
        if re.search(r"[^က-႟\s]", text):
            return "Please only put Burmese text"
    
        # Tokenize the text
        tokenized_text = np.array(tokenizer.encode(tokenizer.tokenize_text(text)))
        # print("token len : ", len(tokenized_text))
        
        # # wrap it in a list
        if not isinstance(tokenized_text, list):
            tokenized_text = [tokenized_text]  # Make sure it's a list
        
        # Pad the tokenized text to make sure it's the correct length (maxlen=300 )
        return pad_sequences(tokenized_text, maxlen=300, padding='post', truncating='post', dtype='int32')
    
    except Exception as e:
        return f"Error : {str(e)}"

def get_prediction(processed_text, threshold, model):
    """
    Predicts the class of the input text using the trained model.
    
    Parameters:
    - processed_text (np.array): The processed (tokenized and padded) input text.
    - threshold (float): The threshold for classifying as hate speech.
    - model: The trained model for prediction.
    
    Returns:
    - str: The predicted class label.
    """
    class_labels = {0: 'Non-hate speech', 1: 'Hate speech'}

    try:
        # Predict probabilities
        result = model.predict(processed_text)[0]

        # Determine the class based on the threshold
        predicted_class = int(result > threshold)

        return class_labels[predicted_class]
    except Exception as e:
        return f"Error: {str(e)}"
    
def get_random_text():
    """
    give an sentence to test the model
    
    Retruns:
    - str: an sentence to test
    """
    
    text_list = ["ဂရိဒဏ္ဍာရီ သည် ရှေးခေတ်ဂရိလူမျိုးများ မူလအနေဖြင့် ပြောဆိုခဲ့ကြသော ဒဏ္ဍာရီ အစုအဝေးတစ်ခုဖြစ်၍ ရှေးဂရိရိုးရာပုံပြင်ဇာတ်လမ်း အမျိုးအစားတစ်ခုလည်းဖြစ်သည်။ ဤဇာတ်လမ်းပုံပြင်များတွင် ကမ္ဘာလောက၏ မူလအစနှင့် သဘောသဘာဝ၊ နတ်ဘုရားများ၊ သူရဲကောင်းများ၊ ဒဏ္ဍာရီလာသတ္တဝါများ စသည်တို့၏ ဘဝနှင့် ဆောင်ရွက်မှုများ၊ ရှေးခေတ်ဂရိလူမျိုးတို့၏ ကိုယ်ပိုင်ယုံကြည်ကိုးကွယ်မှုနှင့် ရိုးရာဓလေ့ကျင့်ထုံးတို့၏ မူလဇာစ်မြစ်များနှင့် အရေးပါမှုတို့ ပါဝင်ကြသည်။",
                 "သင်ခန်းစာလေးတွေ ကြည့်ဖြစ်ကြရဲ့လား ခင်ဗျ",
                 "ဒီခွေးသတေါင်းစါးကတမျိုးနင်တို့ခွေးမျိုးခွေးအုပ်စုရှိရာသွားအူနေစမ်းပါ",
                 "အမှန်တွေပြောရင် မခံနိူင်ဘူးလား ဖင်ခံလိုက်"]

    return random.choice(text_list)
        