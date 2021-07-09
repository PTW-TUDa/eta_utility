from opcua import Client as BaseClient


class Client(BaseClient):
    class MockNode:
        def get_value(self) -> float:
            return 2858.0

    @staticmethod
    def connect() -> None:
        pass

    @classmethod
    def get_node(cls, node) -> MockNode:
        return cls.MockNode()
