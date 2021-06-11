from opcua import Client as OPCUAClient


class Client(OPCUAClient):
    class MockNode:
        def get_value(self) -> float:
            return 2858.0

    @staticmethod
    def connect() -> None:
        pass

    @classmethod
    def get_node(cls, node) -> MockNode:
        return cls.MockNode()
