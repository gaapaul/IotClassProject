import base64

from google.api_core.exceptions import AlreadyExists
from google.cloud import iot_v1
from google.cloud import pubsub
from google.oauth2 import service_account
from googleapiclient import discovery
from googleapiclient.errors import HttpError

def set_config(project_id, cloud_region, registry_id, device_id, config):
	print('Set device configuration')
	client = iot_v1.DeviceManagerClient()
	device_path = client.device_path(
    project_id, cloud_region, registry_id,device_id)

	data = config.encode('utf-8')
	version = 0 #default version
	return client.modify_cloud_to_device_config(device_path, data, version)

def echo_pubsub(event,context):
    message = event['data']
    message = base64.b64decode(message).decode('utf-8')
    deviceId = event['attributes']['deviceId']
    deviceNumId = event['attributes']['deviceNumId']
    deviceRegistryLocation= event['attributes']['deviceRegistryLocation']
    deviceRegistryId= event['attributes']['deviceRegistryId']
    projectId = event['attributes']['projectId']
	payload = json.loads(message)
    msg = payload['MessageSent']
    recipient = payload['To']
    fromUser = payload['From']
    set_config(projectId,deviceRegistryLocation,deviceRegistryId,recipient,message)
set_config