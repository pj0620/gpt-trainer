from openai import OpenAI


class Chat:
    def __init__(self,
                 job_description: str = None,
                 chatgpt_model: str = "gpt-3.5-turbo"):
        self.job_description = job_description
        self.chatgpt_model = chatgpt_model

        self.client = OpenAI()

    def get_response(self,
                     text_prompt: str = None,
                     encoded_image: str = None):
        if (encoded_image is not None) and (not self.image_compatible):
            raise Exception("chat not configured to accept images")

        content = []

        if text_prompt is not None:
            content.append({
                "type": "text",
                "text": text_prompt
            })

        if encoded_image is not None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })

        completion = self.client.chat.completions.create(
            model=self.chatgpt_model,
            messages=[
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": self.job_description
                    }]
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500
        )

        print(f"{completion=}")
