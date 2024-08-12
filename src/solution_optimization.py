import textgrad as tg

tg.set_backward_engine(tg.get_engine("gpt-4o"))

initial_solution = """
To solve the equation 3x^2 - 7x + 2 = 0,
we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)
"""

solution = tg.Variable(
    initial_solution,
    requires_grad=True,
    role_description="solution to the math question",
)

loss_system_prompt = tg.Variable(
    """You will evaluate a solution to a math question.
Do not attempt to solve it yourself, do not give a solution,
only identify errors. Be super concise.""",
    requires_grad=False,
    role_description="system prompt",
)

loss_fn = tg.TextLoss(loss_system_prompt)

optimizer = tg.TGD([solution])

loss = loss_fn(solution)
loss

loss.backward()
optimizer.step()
print(solution.value)
