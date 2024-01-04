import os
import re
import requests
import sys
import numpy as np
import tensorflow as tf
from traceback import format_exc


from unit_cell_tools import calculate_uc_from_s6

def get_current_model(lattice_type, models_path):
    #list all modes in os.path.join(models_path, lattice_type)
    # find and return the most recent model
    print("Not implemented yet")
    pass

def get_prediction(model, peak_list, num_predictions=5):
    try:
        peaks_list = tf.constant(peak_list)
        peaks_list = tf.cast(peaks_list,tf.float32)
        peaks_list = tf.slice(peaks_list,[0],[20])
        peaks_list = tf.reshape(peaks_list,[1,20])
        predictions_list = []
        for i in range(num_predictions):
            preds =  model.predict(peaks_list)
            preds = tf.reshape(preds,[6])
            print(preds)
            if len(preds) == 6:
                predictions_list.append(preds)
        return predictions_list        
    except (ValueError, KeyError, OSError):
        print(format_exc())
        return format_exc()    

def process_predictions(predictions_list):
    uc_param_predictions = []
    uc_params = calculate_uc_from_s6(predictions_list[0].numpy())
    uc_param_predictions.append(uc_params)
    return uc_param_predictions        

if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    #parser.add_argument("--lattice-type", default="oP", type=str)
    parser.add_argument("--model-path", type=Path, default=None, required=True, help="Full path where model is located")
    #parser.add_argument("--url", type=str, default=None,  help="Make a prediction via url")
    #parser.add_argument("--models-dir", type=Path, default="~/repos/MLCell/unit_cell_regression_models/",  help="Path to all models")
    parser.add_argument("--peak-positions", nargs='+', help='ordered list of Bragg peak positions in INVERSE d-space')
    parser.add_argument("--file", type=Path, help="text file containing Bragg peak positions (one per line) in INVERSE d-space")

    args, _ = parser.parse_known_args()

    if args.url:
        pass
        # this could look something like this
        # r = requests.get(args.url+"/my_expected_payload.json")
        # input_json =  r.json()
        # out_url = args.url+"/predictions.html"
        # lattice_type = input_json["lattice_type"]
        # data = input_json["peak_list"]
        # peaks_list = [float(x) for x in data]
        # if  len(peak_list)>19:
        #    out_msg = (f"Need to input a list of at least 20 Bragg peak positions in d-space. Received {len(peaks_list)} peaks")
        #    out_msg = {'estimated_unit_cell_params': "prediction failed: {output}")}
        #    x = requests.post(out_url, json=out_msg)
        # else:
        #     model = get_current model(lattice_type, args.models_dir) 
        #     output = get_prediction(model, peak_list) 
        #     if len(output) > 0:
        #         outdata = process_predictions(output)
        #         if len(outdata) > 0:
        #             outstring_list = []
        #             for uc_pred in outdata:
        #                 uc_string = ",".join(uc_pred)
        #                 outstring_list.append(uc_string)
        #            outdata = {'estimated_unit_cell_params': "/".join(outstring_list)}
        #            x = requests.post(out_url, json=outdata)
        #            print(x.text)
        #     else:     
        #         out_msg = {'estimated_unit_cell_params': "prediction failed: {output}")}
        #         x = requests.post(out_url, json=out_msg)
        #         print(x.text)
              
    elif args.file:   
        with open(args.file, 'r') as in_file:
            file_data = in_file.readlines()
            data = []
            for line in file_data:
                data.append(float(line.strip()))
            
    elif args.peak_positions:   
        data = args.peak_positions
    else:
        print("Need to input an ordered list of at least 20 Bragg peak positions in INVERSE d-space")

    peaks_list = [float(x) for x in data]
    assert len(peaks_list)>19, f"Need to input an ordered list of at least 20 Bragg peak positions in INVERSE d-space. Received {len(peaks_list)} peaks"
    if args.model_path:
        model = tf.keras.models.load_model(args.model_path)
        output = get_prediction(model, peaks_list) 
        if len(output) > 0:
            outdata = process_predictions(output)
            if len(outdata) > 0:
                for uc_pred in outdata:
                    print("Unit cell parameter predictions are ordered by numerical value first by length, then by angle")
                    print("Unit Cell Parameter Predictions:", uc_pred) 
    else:
        model = get_current_model(lattice_type, args.models_dir) 
        output = get_prediction(model, peaks_list) 
        if len(output) > 0:
            outdata = process_predictions(output)
            if len(outdata) > 0:
                for uc_pred in outdata:
                    print("Unit cell parameter predictions are ordered by numerical value first by length, then by angle")
                    print("Unit Cell Parameter Predictions:", uc_pred) 
