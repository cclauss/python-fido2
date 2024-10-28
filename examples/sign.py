# Copyright (c) 2024 Yubico AB
# All rights reserved.
#
#   Redistribution and use in source and binary forms, with or
#   without modification, are permitted provided that the following
#   conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Connects to the first FIDO device found which supports the PRF extension,
creates a new credential for it with the extension enabled, and uses it to
derive two separate secrets.
"""
from fido2 import cbor
from fido2.hid import CtapHidDevice
from fido2.server import Fido2Server
from fido2.client import Fido2Client, WindowsClient, UserInteraction
from fido2.utils import sha256, websafe_encode, websafe_decode
from fido2.cose import CoseKey, ES256
from getpass import getpass
import ctypes
import sys

try:
    from fido2.pcsc import CtapPcscDevice
except ImportError:
    CtapPcscDevice = None


def enumerate_devices():
    for dev in CtapHidDevice.list_devices():
        yield dev
    if CtapPcscDevice:
        for dev in CtapPcscDevice.list_devices():
            yield dev


# Handle user interaction
class CliInteraction(UserInteraction):
    def prompt_up(self):
        print("\nTouch your authenticator device now...\n")

    def request_pin(self, permissions, rd_id):
        return getpass("Enter PIN: ")

    def request_uv(self, permissions, rd_id):
        print("User Verification required.")
        return True


uv = "discouraged"
rk = "discouraged"

if WindowsClient.is_available() and not ctypes.windll.shell32.IsUserAnAdmin():
    # Use the Windows WebAuthn API if available, and we're not running as admin
    client = WindowsClient("https://example.com")
else:
    # Locate a device
    for dev in enumerate_devices():
        client = Fido2Client(
            dev,
            "https://example.com",
            user_interaction=CliInteraction(),
        )
        if "sign" in client.info.extensions:
            break
    else:
        print("No Authenticator with the sign extension found!")
        sys.exit(1)

server = Fido2Server({"id": "example.com", "name": "Example RP"}, attestation="none")
user = {"id": b"user_id", "name": "A. User"}

# Prepare parameters for makeCredential
create_options, state = server.register_begin(
    user,
    resident_key_requirement=rk,
    user_verification=uv,
    authenticator_attachment="cross-platform",
)

message = b"I am a message"
ph_data = sha256(message)

# Create a credential
result = client.make_credential(
    {
        **create_options["publicKey"],
        "extensions": {
            "sign": {
                "generateKey": {"algorithms": [ES256.ALGORITHM], "phData": ph_data}
            }
        },
    }
)

# Complete registration
auth_data = server.register_complete(
    state, result.client_data, result.attestation_object
)
credentials = [auth_data.credential_data]

# PRF result:
sign_result = result.extension_results.get("sign")
print("CREATE sign result", sign_result)
sign_key = sign_result.get("generatedKey")
if not sign_key:
    print("Failed to create credential with sign extension", result.extension_results)
    sys.exit(1)
print("New credential created, with the sign extension.")

pk = CoseKey.parse(
    cbor.decode(websafe_decode(sign_key["publicKey"]))
)  # COSE key in bytes
kh = sign_key["keyHandle"]  # key handle in bytes
signature = sign_result.get("signature")
print("public key", pk)
print("keyHandle", kh)

print("Test verify signature", signature)
pk.verify(message, websafe_decode(signature))
print("Signature verified!")

message = b"New message"
ph_data = sha256(message)

# Prepare parameters for getAssertion
request_options, state = server.authenticate_begin(credentials, user_verification=uv)

# Authenticate the credential
result = client.get_assertion(
    {
        **request_options["publicKey"],
        "extensions": {
            "sign": {
                "sign": {
                    "phData": ph_data,
                    "keyHandleByCredential": {
                        websafe_encode(credentials[0].credential_id): kh,
                    },
                },
            }
        },
    }
)

# Only one cred in allowCredentials, only one response.
result = result.get_response(0)

sign_result = result.extension_results["sign"]
print("GET sign result", sign_result)

signature = sign_result.get("signature")
print("Test verify signature", signature)

pk.verify(message, websafe_decode(signature))
print("Signature verified!")