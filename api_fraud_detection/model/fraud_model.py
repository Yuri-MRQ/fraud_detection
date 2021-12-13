import numpy as np
import pandas as pd

from joblib import load

from tensorflow.keras.models import load_model

def slice_data_for_models(df):

    """
    This function split the df in 2, one with 10 features to be used on
    the random florest model only and other with 7 features to be used
    on the encoder-decoder, them on the RFC
    """

    rf_only_df = df

    columns = ['device_id', 'balance','age_range', 
               'time_client', 'cash_out_type_1',
            'cash_out_type_2', 'cash_out_type_3']

    encoder_df = df[columns]

    return rf_only_df, encoder_df

def predict_vote(predict_encoded, predict_rf_only):

    predict_concat = np.concatenate((np.expand_dims(predict_encoded.T, axis=1),np.expand_dims(predict_rf_only.T, axis=1)*2), axis=1)


    predict = (predict_concat.sum(axis=1)/2).astype(int)

    return predict

def predict(features:dict):

    #Random Forest onlye
    clf_rf_only = load("model/saved_models/rf_classifier/baseline_rfc.joblib")

    #Econder + Random Forest
    rfc_encoder = load("model/saved_models/rf_classifier/model_rfc.joblib")
    encoder  = load_model("model/saved_models/bottleneck")

    features_list = [sample.dict() for sample in features]
    features_df = pd.DataFrame(features_list)

    rf_only_df, encoder_df = slice_data_for_models(features_df)

    x_rf_only = rf_only_df.to_numpy()
    x_encoder_df = encoder_df.to_numpy()


    encoded_x = encoder.predict(x_encoder_df)

    predict_encoded = rfc_encoder.predict(encoded_x)
    predict_rf_only = clf_rf_only.predict(x_rf_only)

    predict = predict_vote(predict_encoded, predict_rf_only)

    predict = predict.tolist()
    predict = dict(enumerate(predict, 1))

    return predict