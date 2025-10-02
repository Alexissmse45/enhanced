import numpy as np
from tensorflow.keras.preprocessing import image
from model.cnn_model import build_cnn_model, SpatialTransformer, HierarchicalAttention
from prediction.advice import get_advice

# -------------------------------
# Step 1: Load categories dynamically
# -------------------------------
with open("categories.txt") as f:
    CATEGORIES = [line.strip() for line in f]

# -------------------------------
# Step 2: Context-aware rules (text fusion)
# -------------------------------
TEXT_RULES = {
    "bleeding": ["cut","bleeding","blood","gash"],
    "burn": ["burn","scald","fire","hot"],
    "fracture": ["broken","fracture","sprain","crack"],
    "choking": ["choke","choking","airway","can't breathe", "breathe", "cough", "gag","breath", "throat","swallow", "breathing"],
    "allergic reaction": ["allergy","rash","swelling","hives", "red eyes", "itchy", "redness"],
    "dog bite": ["dog bite","dog"],
    "cat claw": ["cat scratch","cat","claw"],
    "mosquito bite": ["mosquito","itch","red bump"],
    "electrical injuries": ["electric","shock","electrocute","current"],
    "frostbite": ["frostbite","cold","frozen","numb","blue skin","ice","snow","hypothermia", "black" ],
}

# -------------------------------
# Step 3: Predict from image
# -------------------------------
def predict_image(model, img_path, text_hint=None):
    # Preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    # CNN prediction
    probs = model.predict(arr)[0]
    img_idx = int(np.argmax(probs))
    img_pred = CATEGORIES[img_idx]
    img_conf = float(probs[img_idx])

    # -------------------------------
    # Step 4: Context-aware fusion if text provided
    # -------------------------------
    if text_hint:
        txt = text_hint.lower()
        matches = []  # collect multiple possible injuries from text
        for k, v in TEXT_RULES.items():
            if any(w in txt for w in v):
                matches.append(k)

        if matches:
            fused = 0.7 * probs  # keep CNN weight
            text_probs = np.zeros_like(probs)
            for label in matches:  # boost all matched injuries
                try:
                    tidx = CATEGORIES.index(label)
                    text_probs[tidx] = 1.0
                except ValueError:
                    continue
            fused += 0.3 * text_probs
            fused /= np.sum(fused)

            final_idx = int(np.argmax(fused))
            final_label = CATEGORIES[final_idx]
            final_conf = float(fused[final_idx])

            # Step 5: If multiple matches, return all advice
            advice_texts = []
            for label in matches:
                advice_texts.append(get_advice(label, 0.9))
            if final_label not in matches:
                advice_texts.append(get_advice(final_label, final_conf))

            return final_label, final_conf, advice_texts
        else:
            # No text match, fallback to CNN
            final_label, final_conf = img_pred, img_conf
            advice_text = get_advice(final_label, final_conf)
            return final_label, final_conf, [advice_text]

    else:
        # Image-only prediction
        final_label, final_conf = img_pred, img_conf
        advice_text = get_advice(final_label, final_conf)
        return final_label, final_conf, [advice_text]

# -------------------------------
# Step 6: Predict from text input only
# -------------------------------
def predict_text(user_text):
    txt = user_text.lower()
    matches = []

    for k, v in TEXT_RULES.items():
        if any(w in txt for w in v):
            matches.append(k)

    if matches:
        return matches, 0.9
    return ["unknown"], 0.3

# -------------------------------
# Step 7: Confidence reinforcement
# -------------------------------
def adjust_confidence(final_label, final_conf, is_correct):
    if is_correct:
        # Boost confidence slightly if confirmed correct
        final_conf = min(1.0, final_conf + 0.05)
    else:
        # Drop confidence if confirmed wrong
        final_conf = max(0.0, final_conf * 0.7)
    return final_conf