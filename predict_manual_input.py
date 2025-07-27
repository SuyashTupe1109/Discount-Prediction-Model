import joblib
import pandas as pd

def predict_discount():
    try:
        
        model = joblib.load('myntra_discount_model.pkl')
        
        required_features = model.feature_names_in_
        print(f"Model requires these features in order: {list(required_features)}")
        
        input_data = {}
        print("\n=== Discount Prediction Tool ===")
        
        feature_prompts = {
            'OriginalPrice (in Rs)': "Enter Original Price (₹): ",
            'Brand_importance': "Enter Brand Importance (0-1000 scale): ",
            'ind_cat_popularity': "Enter Sub-Category Popularity (# of brands): ",
            'cat_popularity': "Enter Main Category Popularity (# of products): ",
            'gender': "Enter Gender (0=Men, 1=Women/Unisex): "
        }
        
        for feature in required_features:
            try:
                value = float(input(feature_prompts[feature]))
                if feature == 'gender': 
                    value = int(value)
                input_data[feature] = value
            except ValueError:
                print(f"Error: Invalid input for {feature}. Please enter numbers only.")
                return

        
        input_df = pd.DataFrame([input_data])[required_features]
        
        prediction = model.predict(input_df)
        
        price = input_data['OriginalPrice (in Rs)']
        discount_pct = prediction[0]
        discounted_price = price * (1 - discount_pct/100)
        
        print(f"\nPredicted Discount: {discount_pct:.1f}%")
        print(f"Suggested Discounted Price: ₹{discounted_price:,.2f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Common fixes:")
        print("- Verify model file exists and isn't corrupted")
        print("- Check all required features are provided")
        print("- Ensure Python environment matches training environment")

if __name__ == "__main__":
    predict_discount()