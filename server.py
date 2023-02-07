from flask import Flask, render_template, request
import pickle
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
import numpy as np

covtype = fetch_covtype()
model = pickle.load(open("./random forest.pkl", "rb"))

result = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen, Douglas-fir",
    5: "Krummholz",
}


def convertToDf(data):
    df = pd.DataFrame(data, columns=covtype.feature_names, index=[0])
    pd.get_dummies(
        df,
        columns=[
            "Wilderness_Area_0",
            "Wilderness_Area_1",
            "Wilderness_Area_2",
            "Wilderness_Area_3",
            "Soil_Type_0",
            "Soil_Type_1",
            "Soil_Type_2",
            "Soil_Type_3",
            "Soil_Type_4",
            "Soil_Type_5",
            "Soil_Type_6",
            "Soil_Type_7",
            "Soil_Type_8",
            "Soil_Type_9",
            "Soil_Type_10",
            "Soil_Type_11",
            "Soil_Type_12",
            "Soil_Type_13",
            "Soil_Type_14",
            "Soil_Type_15",
            "Soil_Type_16",
            "Soil_Type_17",
            "Soil_Type_18",
            "Soil_Type_19",
            "Soil_Type_20",
            "Soil_Type_21",
            "Soil_Type_22",
            "Soil_Type_23",
            "Soil_Type_24",
            "Soil_Type_25",
            "Soil_Type_26",
            "Soil_Type_27",
            "Soil_Type_28",
            "Soil_Type_29",
            "Soil_Type_30",
            "Soil_Type_31",
            "Soil_Type_32",
            "Soil_Type_33",
            "Soil_Type_34",
            "Soil_Type_35",
            "Soil_Type_36",
            "Soil_Type_37",
            "Soil_Type_38",
            "Soil_Type_39",
        ],
    )
    return df


def convertToTestData(
    elevation,
    aspect,
    slope,
    horizontal_distance_hydrology,
    vertical_distance_hydrology,
    horizontal_distance_roadways,
    hillshade_9am,
    hillshade_noon,
    hillshade_3pm,
    horizontal_distance_fire_points,
):
    test_data_dict = {
        "Elevation": elevation,
        "Aspect": aspect,
        "Slope": slope,
        "Horizontal_Distance_To_Hydrology": horizontal_distance_hydrology,
        "Vertical_Distance_To_Hydrology": vertical_distance_hydrology,
        "Horizontal_Distance_To_Roadways": horizontal_distance_roadways,
        "Hillshade_9am": hillshade_9am,
        "Hillshade_Noon": hillshade_noon,
        "Hillshade_3pm": hillshade_3pm,
        "Horizontal_Distance_To_Fire_Points": horizontal_distance_fire_points,
        "Wilderness_Area_0": 1,
        "Wilderness_Area_1": 0,
        "Wilderness_Area_2": 0,
        "Wilderness_Area_3": 0,
        "Soil_Type_0": 1,
        "Soil_Type_1": 0,
        "Soil_Type_2": 0,
        "Soil_Type_3": 0,
        "Soil_Type_4": 0,
        "Soil_Type_5": 0,
        "Soil_Type_6": 0,
        "Soil_Type_7": 0,
        "Soil_Type_8": 0,
        "Soil_Type_9": 0,
        "Soil_Type_10": 0,
        "Soil_Type_11": 0,
        "Soil_Type_12": 0,
        "Soil_Type_13": 0,
        "Soil_Type_14": 0,
        "Soil_Type_15": 0,
        "Soil_Type_16": 0,
        "Soil_Type_17": 0,
        "Soil_Type_18": 0,
        "Soil_Type_19": 0,
        "Soil_Type_20": 0,
        "Soil_Type_21": 0,
        "Soil_Type_22": 0,
        "Soil_Type_23": 0,
        "Soil_Type_24": 0,
        "Soil_Type_25": 0,
        "Soil_Type_26": 0,
        "Soil_Type_27": 0,
        "Soil_Type_28": 0,
        "Soil_Type_29": 0,
        "Soil_Type_30": 0,
        "Soil_Type_31": 0,
        "Soil_Type_32": 0,
        "Soil_Type_33": 0,
        "Soil_Type_34": 0,
        "Soil_Type_35": 0,
        "Soil_Type_36": 0,
        "Soil_Type_37": 0,
        "Soil_Type_38": 0,
        "Soil_Type_39": 0,
    }
    return test_data_dict

def predict(prediction_data_dict):
    prediction_df = convertToDf(prediction_data_dict)
    prediction = model.predict(prediction_df)
    return result[prediction[0]]


app=Flask(__name__)

@app.route("/predict", methods=['GET'])
def index():
    print(request.args)
    elevation = int(request.args.get('elevation'))
    aspect = int(request.args.get('aspect'))
    slope = int(request.args.get('slope'))
    horizontal_distance_to_hydrology = int(request.args.get('horizontal_distance_hydrology'))
    vertical_distance_to_hydrology = int(request.args.get('vertical_distance_hydrology'))
    horizontal_distance_to_roadways = int(request.args.get('horizontal_distance_roadways'))
    hillshade_9am = int(request.args.get('hillshade_9am'))
    hillshade_noon = int(request.args.get('hillshade_noon'))
    hillshade_3pm = int(request.args.get('hillshade_3pm'))
    horizontal_distance_to_fire_points = int(request.args.get('horizontal_distance_fire_points'))

    # Convert to test data
    test_data = convertToTestData(
        elevation=elevation,
        aspect=aspect,
        slope=slope,
        horizontal_distance_hydrology=horizontal_distance_to_hydrology,
        vertical_distance_hydrology=vertical_distance_to_hydrology,
        horizontal_distance_roadways=horizontal_distance_to_roadways,
        hillshade_9am=hillshade_9am,
        hillshade_noon=hillshade_noon,
        hillshade_3pm=hillshade_3pm,
        horizontal_distance_fire_points=horizontal_distance_to_fire_points,
    )
    print(test_data)
    prediction_df = convertToDf(test_data)
    prediction = predict(prediction_df)
    return prediction


if __name__ == '__main__':
    app.run(debug=True)
    