import json
import base64

from google.cloud import iot_v1
from google.cloud import pubsub

def set_config(project_id, cloud_region, registry_id, device_id, config):
    client = iot_v1.PublisherClient()
    device_path = client.device_path(
    project_id, cloud_region, registry_id,device_id)
    data = config.encode('utf-8')
    version = 0 #default version
    return client.modify_cloud_to_device_config(device_path, data, version)

def send_message(project_id, cloud_region, registry_id, device_id, message):
    client = iot_v1.DeviceManagerClient()
    device_path = client.device_path(
    project_id, cloud_region, registry_id, device_id)
    data = message.encode('utf-8')
    version = 0 #default version
    return client.send_command_to_device(device_path, data, version)

def echo_pubsub1(event,context):
    #Function Executes when topic is updated
    message_event = event['data']
    message = base64.b64decode(message_event).decode('utf-8')
    deviceRegistryLocation= event['attributes']['deviceRegistryLocation']
    deviceRegistryId= event['attributes']['deviceRegistryId']
    projectId = event['attributes']['projectId']
    payload = json.loads(message)
    #gather info and send message to device
    if(payload['Type'] == 0): #data
        recipient = payload['To']
        my_message = message
        send_message(projectId,deviceRegistryLocation,deviceRegistryId,recipient,str(my_message))
