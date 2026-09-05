"""Minimal SDK example; set QIONGYU_BASE_URL and QIONGYU_API_TOKEN first."""

from qiongyu_sdk import QiongyuClient


def main() -> None:
    client = QiongyuClient()
    print(client.health())
    upload = client.upload("pollution.csv", "weather.csv")
    result = client.predict(upload["session_id"])
    print(result)
    client.download(upload["session_id"], "predictions.csv")


if __name__ == "__main__":
    main()
