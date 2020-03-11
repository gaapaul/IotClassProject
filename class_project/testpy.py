import json
import base64

# from google.api_core.exceptions import AlreadyExists
# from google.cloud import iot_v1
# from google.cloud import pubsub
# from google.oauth2 import service_account
# from googleapiclient import discovery
# from googleapiclient.errors import HttpError

# def set_config(project_id, cloud_region, registry_id, device_id, config):
# 	print('Set device configuration')
# 	client = iot_v1.DeviceManagerClient()
# 	device_path = client.device_path(
#     project_id, cloud_region, registry_id,device_id)

# 	data = config.encode('utf-8')

# 	return client.modify_cloud_to_device_config(device_path, data, version)
event = {'@type': 'type.googleapis.com/google.pubsub.v1.PubsubMessage', 'attributes': {'deviceId': 'test-dev2', 'deviceNumId': '2636914494162648', 'deviceRegistryId': 'tour-registry', 'deviceRegistryLocation': 'us-central1', 'projectId': 'turnkey-banner-265721', 'subFolder': ''}, 'data': 'eyJ0ZW1wZXJhdHVyZSI6IDN9'}

message = event['data']
message = base64.b64decode(message).decode('utf-8')
deviceId = event['attributes']['deviceId']
deviceNumId = event['attributes']['deviceNumId']
deviceRegistryLocation= event['attributes']['deviceRegistryLocation']
deviceRegistryId= event['attributes']['deviceRegistryId']
projectId = event['attributes']['projectId']

print("Send Message:")
usrInputMsg=input()
print("To?:")
usrInputAddr=input()
# Report the device's temperature to the server by serializing it
# as a JSON string.
payload = json.dumps({'MessageSent': usrInputMsg, 'To' : usrInputAddr, 'From' : deviceId})
payload = json.loads(payload)
msg = payload['MessageSent']
recipient = payload['To']
fromUser = payload['From']
print(msg)
print(recipient)
print(fromUser)








#set_config(projectId,deviceRegistryLocation,deviceRegistryId,deviceId,message)
