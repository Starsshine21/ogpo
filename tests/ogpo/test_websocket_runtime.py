from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class _FakeConnection:
    def recv(self):
        return msgpack_numpy.Packer().pack({})


def test_websocket_client_disables_keepalive_for_long_model_inference(monkeypatch):
    captured = {}

    def fake_connect(uri, **kwargs):
        captured.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", fake_connect)

    websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)

    assert captured["ping_interval"] is None
