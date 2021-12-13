# Fraud detection system

This is a fraud detection system that use an ensemble model e a light API.

First model:

The encoder-decoder DNN has the input vector have the size (sample, 10) and the output (sample, 10), the loss used was
MES (mean squared error), the encoded vector (ʋ) have shape (sample, 4).
The architecture of the model have 03 Dense layers as hidden layer and follow this architecture:

In this [file](data_science_challenge_dataset/encoder_decoder.ipynb)
is the encoder-decoder model.

With the ʋ we predicts the target value using a Random Forest Classifier, the hypeparameters used
are in this [file](data_science_challenge_dataset/rencoder-rf.ipynb)

Second Model:

Is a simple [Random Forest Classifier](data_science_challenge_dataset/random_florest_model.ipynb)

And in the final predict the Random Forest model has weight 02 the formula used to the final predict was:

![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7Brf%2A2%2Bencoder%7D%7B2%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

For the API we used the FastAPI. On the localhost/predict we send a header with the values to predict
the target, the features necessary to predict are:

- device_id
- balance
- processed_at
- is_fraud
- age_range
- number_of_selfies_sent
- time_client (is the processed_at minus the first entry of the account on the database)
- cash_out_type_1
- cash_out_type_2
- cash_out_type_3
- cash_out_type_6

## To run

OS X & Linux:

```
git clone https://github.com/Yuri-MRQ/fraud_detection.git

cd api_fraud_detection

pip install -r requeriment.txt

uvicorn main:app --reload

```

## Testing

Once the API is runing go to api_fraud_detection/test and run [test_api.ipynb](api_fraud_detection/test/test_api.ipynb)
