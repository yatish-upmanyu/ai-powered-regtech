from flask import Flask, request, jsonify
import re

app = Flask(__name__)

@app.route('/edorasone/api/process', methods=['POST'])
def handle_process():
    if request.args.get('view') == 'full':
        try:
            data = request.get_json()
            if data:  # Check if data is not None
                url = data.get('url', '')
                match = re.search(r'ratingsystems/(\d+)', url)
                if match:
                    customer_id = match.group(1)
                    response_data = {
                        "modificationCreatorId": "GEAR-6e80ca89-6d77-45fd-98e1-04455949242c",
                        "RestResult": {
                            "response_content": {
                                "customerId": int(customer_id),
                                "scope": {
                                    "bucketCode": "RF",
                                    "bucketDescription": "IP",
                                    "modelSet": "12",
                                    "ratingPrefix": "RF",
                                    "maxRegime": "AIRB",
                                    "pdModelCode": "RF"
                                }
                            },
                            "return_code": 200
                        },
                        "creationTime": "2024-11-27T09:24:43.437Z",
                        "modificationUpdaterId": "GEAR-6e80ca89-6d77-45fd-98e1-04455949242c",
                        "assigneeIdUpdateTime": "2024-11-27T09:24:43.541Z",
                        "ownerId": "GEAR-6e80ca89-6d77-45fd-98e1-04455949242c",
                        "type": "PRC",
                        "content": "Content-Type|application/json",
                        "path": [
                            "GEAR-463bc339-eb87-4444-b452-ede43e4d4ae5"
                        ],
                        "subState": "COMPLETED",
                        "providerId": "ActivitiProcessProvider",
                        "host": "host|api.xyz.com",
                        "id": "GEAR-463bc339-eb87-4444-b452-ede43e4d4ae5",
                        "state": "COMPLETED",
                        "definitionId": "GEAR-b71c4c03-e2cb-417d-8ac4-1a579e287e97",
                        "jsonpayload": "\"\"",
                        "method": "GET",
                        "candidateGroupIds": [],
                        "globalId": "PRC-0ba662cc-00db-49ee-a91c-0e85bbe92b68",
                        "externalId": "ACT-3779186",
                        "modificationVersion": 3,
                        "updateTime": "2024-11-27T09:24:43.953Z",
                        "assigneeId": "GEAR-6e80ca89-6d77-45fd-98e1-04455949242c",
                        "url": f"http://localhost:8456/api/v1/scoping/ratingsystems/{customer_id}",
                        "candidateUserIds": [],
                        "subStateUpdateTime": "2024-11-27T09:24:43.953Z",
                        "initialAssigneeId": "GEAR-6e80ca89-6d77-45fd-98e1-04455949242c",
                        "name": "Risk API",
                        "tenantId": "11111111-1111-1111-1111-111111111111",
                        "stateUpdateTime": "2024-11-27T09:24:43.953Z"
                    }
                    return jsonify(response_data), 200
                else:
                    return jsonify({"error": "CustomerId not found in URL"}), 400
            else:
                return jsonify({"error": "Missing or invalid request data"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "view parameter must be 'full'"}), 400

if __name__ == '__main__':
    app.run(port=8080)