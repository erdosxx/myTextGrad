import io
from PIL import Image

import textgrad as tg

# differently from the past tutorials, we now need a multimodal LLM call instead of a standard one!
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss

tg.set_backward_engine("gpt-4o")

import httpx

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"

image_data = httpx.get(image_url).content

image_variable = tg.Variable(
    image_data,
    role_description="image to answer a question about",
    requires_grad=False,
)

question_variable = tg.Variable(
    "What do you see in this image?",
    role_description="question",
    requires_grad=False,
)
response = MultimodalLLMCall("gpt-4o")([image_variable, question_variable])
response

loss_fn = ImageQALoss(
    evaluation_instruction="Does this seem like a complete and good answer for the image? Criticize. Do not provide a new answer.",
    engine="gpt-4o",
)
loss = loss_fn(
    question=question_variable, image=image_variable, response=response
)
loss

optimizer = tg.TGD(parameters=[response])
loss.backward()
optimizer.step()
print(response.value)
