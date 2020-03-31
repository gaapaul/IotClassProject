# Copyright 2017 Google Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Code edited from end to end example Github link below
#https://github.com/GoogleCloudPlatform/python-docs-samples.git

"""
To connect the device you must have downloaded Google's CA root certificates,
and a copy of your private key file. See cloud.google.com/iot for instructions
on how to do this. Run this script with the corresponding algorithm flag.

  $ python cloudiot_pubsub_example_mqtt_device.py \
      --project_id=my-project-id \
      --registry_id=example-my-registry-id \
      --device_id=my-device-id \
      --private_key_file=rsa_private.pem \
      --algorithm=RS256
"""
import argparse
import datetime
import json
import os
import ssl
import time
import base64
import jwt
import paho.mqtt.client as mqtt

from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials

import argparse
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def upload_blob(bucket_name, blob_text, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(blob_text)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def log_data(request):
    request_json = request.get_json()
    BUCKET_NAME = 'iot_bucket_453'
    BLOB_NAME = 'test-blob'
    BLOB_STR = '{"blob": "some json"}'

    upload_blob(BUCKET_NAME, BLOB_STR, BLOB_NAME)
    return

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def create_jwt(project_id, private_key_file, algorithm):
    """Create a JWT (https://jwt.io) to establish an MQTT connection."""
    token = {
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
        'aud': project_id
    }
    with open(private_key_file, 'r') as f:
        private_key = f.read()
    print('Creating JWT using {} from private key file {}'.format(
        algorithm, private_key_file))
    return jwt.encode(token, private_key, algorithm=algorithm)


def error_str(rc):
    """Convert a Paho error to a human readable string."""
    return '{}: {}'.format(rc, mqtt.error_string(rc))


class Device(object):
    """Represents the state of a single device."""

    def __init__(self):
        self.connected = False
        self.start_predict = False
        self.save_data = False
        self.reset = False
        self.temps = np.zeros((1,360,2)) #two features air and out temp
        self.Run_ID = ""
    def log_air_temp(self,index,air,time):
        self.temps[0,index,0] = air
        self.temps[0,index,1] = time
    def read_air_temp(self):
        return self.temps[0,:,0]
    def reset_false(self):
        self.reset = False
    def read_reset(self):
        return self.reset
    def start_predict(self):
        self.start_predict = True
        print("Start Predict On")
    def stop_predict(self):
        self.start_predict = False
        print("Stop Predict")
    def read_predict_state(self):
        return self.start_predict
    def read_data_state(self):
        return self.save_data
    def stop_save_data_state(self):
        self.save_data = False
    def get_Run_ID(self):
        return self.Run_ID
    # def write_data_to_np(self):
    #     for i in range(0, len(self.store_data[:])):
    #         for j in range(0, len(sself.tore_data[0][:]),2):
    #             self.temps[0,len(self.store_data[:])*(i) + int(j/2),0] = self.store_data[i][j]
    #             self.temps[0,len(self.store_data[:])*(i) + int(j/2),1] = self.store_data[i][j+1]
    #     self.store_data = [[]]
    #     self.time_data = []
    def make_prediction(self):
        #in_file_name = parsed_args.in_file
        scalers = {}
        local_batch_size = 18
        for i in range(len(self.temps[0,:,0])-1,-1,-1):
            print(self.temps[0,i,0])
            index = i
            if(self.temps[0,i,0] != 0):
                temp = self.temps[0,i,0]
                break
        for i in range(index,len(self.temps[0,:,0])):
            self.temps[0,i,0] = temp
            self.temps[0,i,1] = i * 10

        print("temps")
        print(self.temps[0,:,0])
        print("times")
        print(self.temps[0,:,1])
        
        # bucket_name = "iot_bucket_453"
        # download_blob(bucket_name, "model_weights_data2_lstm.h5", "model_weights_data2_lstm.h5")

        model = load_model("model_weights_data2_lstm.h5")    
   
        #val_scaled = self.temps
        val_data_bak = self.temps
        val_scaled = np.zeros((360,1,2))
        #Scale data that change during run this way
        for j in range(val_scaled.shape[0]):    
            scalers[0] = MinMaxScaler(feature_range=(0, 1))
            scalers[1] = MinMaxScaler(feature_range=(0, 1))
            val_scaled[:,0, 0:1] = scalers[0].fit_transform(val_data_bak[0,:,1:2])  
            val_scaled[:,0, 1:2] = scalers[1].fit_transform(val_data_bak[0,:,0:1])
        #print(val_scaled[0:50,:,:])
        print(val_scaled[:,0:1,:].shape)
        yhat = model.predict(val_scaled[:,0:1,:] , batch_size = local_batch_size)
        val_scaler = MinMaxScaler(feature_range=(0,1)).fit(self.temps[0,:,0:1])   
        inv_yhat_out = val_scaler.inverse_transform(yhat)
        print(yhat)
        print(inv_yhat_out)
        stop_time = 360
        for i in range(0, len(inv_yhat_out)):
            if(inv_yhat_out[i] > 98):
                stop_time = self.temps[0,i,1]
                break
                print("Stop Time is: "+str(stop_time))
        return stop_time, inv_yhat_out

    def wait_for_connection(self, timeout):
        """Wait for the device to become connected."""
        total_time = 0
        while not self.connected and total_time < timeout:
            time.sleep(1)
            total_time += 1

        if not self.connected:
            raise RuntimeError('Could not connect to MQTT bridge.')
        
    def on_connect(self, unused_client, unused_userdata, unused_flags, rc):
        """Callback for when a device connects."""
        print('Connection Result:', error_str(rc))
        self.connected = True

    def on_disconnect(self, unused_client, unused_userdata, rc):
        """Callback for when a device disconnects."""
        print('Disconnected:', error_str(rc))
        self.connected = False

    def on_publish(self, unused_client, unused_userdata, unused_mid):
        """Callback when the device receives a PUBACK from the MQTT bridge."""
        print('Published message acked.')

    def on_subscribe(self, unused_client, unused_userdata, unused_mid,
                     granted_qos):
        """Callback when the device receives a SUBACK from the MQTT bridge."""
        print('Subscribed: ', granted_qos)
        if granted_qos[0] == 128:
            print('Subscription failed.')

    def on_message(self, unused_client, unused_userdata, message):
        index = 0
        """Callback when the device receives a message on a subscription."""
        payload = message.payload.decode('utf-8')
        payload_dict = json.loads(payload)
        #print(payload["Data"]["Index"])
	   #print(payload)
        #print("Type: "+str(payload_dict["Type"]))
        #print(payload_dict["Data"])
        if(payload_dict["Type"] == 0): #It's Data 
            #print(payload_dict["Data"]["Index"])
            for i in payload_dict["Data"]["Index"]:
                print(payload_dict["Data"][str(i)]["Temp"])
                self.temps[0,i,0] = float(payload_dict["Data"][str(i)]["Temp"])
                self.temps[0,i,1] = float(payload_dict["Data"][str(i)]["Time"])

        if(payload_dict["Type"] == 1):
            if(payload_dict["Data"] == "Predict"):
                self.start_predict = True
                print("Starting Prediction")
            if(payload_dict["Data"] == "Done"):
                self.save_data = True
                print("Time to save Data")
            if(payload_dict["Data"] == "Reset"):
                self.start_predict = False
                self.save_data = False
                self.temps = np.zeros((1,360,2)) #two features air and out temp
                self.reset = True
            if( "Run_ID"  in payload_dict["Data"]):
                self.Run_ID = payload_dict["Data"]["Run_ID"]
                print("ID set to {}".format(self.Run_ID))
        return

def parse_command_line_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Example Google Cloud IoT MQTT device connection code.')
    parser.add_argument(
        '--project_id',
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        required=True,
        help='GCP cloud project name.')
    parser.add_argument(
        '--registry_id', required=True, help='Cloud IoT registry id')
    parser.add_argument(
        '--device_id',
        required=True,
        help='Cloud IoT device id')
    parser.add_argument(
        '--private_key_file', required=True, help='Path to private key file.')
    parser.add_argument(
        '--algorithm',
        choices=('RS256', 'ES256'),
        required=True,
        help='Which encryption algorithm to use to generate the JWT.')
    parser.add_argument(
        '--cloud_region', default='us-central1', help='GCP cloud region')
    parser.add_argument(
        '--ca_certs',
        default='roots.pem',
        help='CA root certificate. Get from https://pki.google.com/roots.pem')
    parser.add_argument(
        '--num_messages',
        type=int,
        default=100,
        help='Number of messages to publish.')
    parser.add_argument(
        '--mqtt_bridge_hostname',
        default='mqtt.googleapis.com',
        help='MQTT bridge hostname.')
    parser.add_argument(
        '--mqtt_bridge_port', type=int, default=8883, help='MQTT bridge port.')
    parser.add_argument(
        '--message_type', choices=('event', 'state'),
        default='event',
        help=('Indicates whether the message to be published is a '
              'telemetry event or a device state message.'))
    parser.add_argument('-i', '--in_file', help='Input file for Network weights',required=False)

    return parser.parse_args()
def dir_path(string):
    string2 = os.getcwd() + string
    if os.path.isdir(string2):
        return string2
    elif os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def main():
    args = parse_command_line_args()
    # with open('turnkey-banner-265721-da1327341af6.json', 'r') as json_file:
    #     data = json.load(json_file)
    # credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    #     data
    # )

    data_log = 'run0.txt'
    read_log = 'data_2025100.csv' 
    # Create the MQTT client and connect to Cloud IoT.
    client = mqtt.Client(
        client_id='projects/{}/locations/{}/registries/{}/devices/{}'.format(
            args.project_id,
            args.cloud_region,
            args.registry_id,
            args.device_id))
    client.username_pw_set(
        username='unused',
        password=create_jwt(
            args.project_id,
            args.private_key_file,
            args.algorithm))
    client.tls_set(ca_certs=args.ca_certs, tls_version=ssl.PROTOCOL_TLSv1_2)

    device = Device()

    client.on_connect = device.on_connect
    client.on_publish = device.on_publish
    client.on_disconnect = device.on_disconnect
    client.on_subscribe = device.on_subscribe
    client.on_message = device.on_message

    client.connect(args.mqtt_bridge_hostname, args.mqtt_bridge_port)

    client.loop_start()

    # This is the topic that the device will publish telemetry events
    # (temperature data) to.
    mqtt_telemetry_topic = '/devices/{}/events'.format(args.device_id)

    # This is the topic that the device will receive configuration updates on.
    mqtt_config_topic = '/devices/{}/commands/#'.format(args.device_id)

    # Wait up to 5 seconds for the device to connect.
    device.wait_for_connection(5)

    # Subscribe to the config topic.
    client.subscribe(mqtt_config_topic, qos=1)
    input_string = ""
    print("Waiting for data to Store then predict on")
    # Update and publish temperature readings at a rate of one per second.
    start_time = time.time()
    while(True):
        time.sleep(.1)
        if(device.read_predict_state()): #Got config update to turn lamp on.
            start_time = time.time()
            #done_payload = payload = json.dumps({"Type": 1, "Data" : "Thanks for Data", "Time" : str(time.time() - start_time), "To" : "test-dev"})
            #device.write_data_to_np()
            #client.publish(mqtt_telemetry_topic, done_payload, qos=1)
            stop_time, prediction = device.make_prediction()
            with open("yhat_vm.txt", "w") as prediction_file:
                for line in prediction:
                    prediction_file.write(str(line) + "\n")
                    print(line)
                prediction_file.close()
            stop_time = "Stop Time" + str(stop_time)
            print(stop_time)
            stop_payload = json.dumps({"Type": 1, "Data" : stop_time, "Time" : str(time.time() - start_time)[:4], "To" : "test-dev"})
            client.publish(mqtt_telemetry_topic, stop_payload, qos=1)
            device.stop_predict()
        if(device.read_data_state()):
            device.stop_save_data_state()
            done_payload = json.dumps({"Type": 1, "Data" : "Thanks for Data", "Time" : str(time.time() - start_time), "To" : "test-dev"})
            client.publish(mqtt_telemetry_topic, done_payload, qos=1)
            done_payload = json.dumps({"Type": 1, "Data" : "Wait State", "Time" : str(time.time() - start_time), "To" : "test-dev2"})
            client.publish(mqtt_telemetry_topic, done_payload, qos=1)
            storage_client = storage.Client(credentials=credentials, project='turnkey-banner-26a5721')
            bucket = storage_client.get_bucket('iot_bucket_453')
            blob = bucket.blob('yhat_vm_file_{}'.format(device.get_Run_ID()))
            blob.upload_from_filename("yhat_vm.txt")
            print("Return Prediction to User")
            payload = json.dumps({"Type": 0, "Data" : prediction.tolist(), "Time" : str(time.time() - start_time)[:4], 'To' : "test-dev3"}) 
            client.publish(mqtt_telemetry_topic, payload, qos=1)
            air_temp = device.read_air_temp()
            time.sleep(1)
            payload = json.dumps({"Type": 1, "Data" : air_temp.tolist(), "Time" : str(time.time() - start_time)[:4], 'To' : "test-dev3"}) 
            client.publish(mqtt_telemetry_topic, payload, qos=1)
            print("Waiting for data to Store")
            storage_client = storage.Client(credentials=credentials, project='turnkey-banner-265721')
            bucket = storage_client.get_bucket('iot_bucket_453')
            blob = bucket.blob('yhat_vm_file')
            blob.upload_from_filename("yhat_vm.txt")
            print("Saved File")
        if(device.read_reset()):
            print("Reset")
            payload = json.dumps({"Type": 1, "Data" : "Wait State", "Time" : str(time.time() - start_time), "To" : "test-dev2"})
            client.publish(mqtt_telemetry_topic, payload, qos=1)
            payload = json.dumps({"Type": 1, "Data" : "Reset Done 2", "Time" : str(time.time() - start_time), "To" : "test-dev3"})
            client.publish(mqtt_telemetry_topic, payload, qos=1)
            device.reset_false()
    client.disconnect()
    client.loop_stop()
    print('Finished loop successfully. Goodbye!')


if __name__ == '__main__':
    main()
