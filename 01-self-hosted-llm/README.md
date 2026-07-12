# Setting up a self-hosted LLM

In this exercise, we're going to set up our own LLM running on the cloud. We'll be able to grant arbitrary people access to this LLM and can use it from other tools, such as our agentic harness.

We'll be running [Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b) on [Brev](https://brev.nvidia.com/). Qwen3.6-27B is an extremely powerful model for its small size and has been called ["the canonical local model to use Hermes Agent with"](https://old.reddit.com/r/LocalLLaMA/comments/1sz2y76/ama_with_nous_research_ask_us_anything/oiynr8i/) ([Hermes Agent](https://hermes-agent.nousresearch.com/) is an agentic harness focused on always-on behavior and the agent's self-evolution).

## Exercises

Set up a Brev instance running your local AI agent and make it accessible from the outside.

1. Create an account at https://brev.nvidia.com/.
2. Register the credit coupon to obtain compute budget:
   1. Select "Billing" at the top of the website.
   2. In the "Pay For Compute" section, find "2. Credits" and select "Redeem Code".
   3. Enter your code under the "Enter Code" text box and select "Redeem".
3. Using a [launchable](https://brev.nvidia.com/launchables/), or manually, host Qwen3.6-27B via vLLM.

   A [launchable](https://brev.nvidia.com/launchables/) allows you to set the model up with the click of one button. Doing it manually entails you running pre-existing scripts, so you'd learn to access the machine and run commands on it yourself.

   First, the tasks for the launchable:
   1. Check available launchables in the "Launchables" section in this documentation below.
   2. Pick a setup that sounds good and select the appropriate instance type:
      - L40S GPUs use a slightly less exact model and allow for fewer concurrent prompts, but are faster.
      - A100 GPUs are slower, but use the full-precision model and allow for more concurrent prompts.
      - Others GPUs use a slightly less exact model and allow for fewer concurrent prompts when you use the configurations that use only a single GPU. They are faster but generally less likely to be available.
   3. You may have to increase the number of GPUs if machines aren't available by selecting "Edit" next to "Instance type" on the right. Remember to stick with the same GPU type.
   4. Optionally, in "Setup values" on the right, set a `VLLM_API_KEY` to secure your agent with an API key that you define.
   5. Optionally, in "Setup values" on the right, set a `HF_TOKEN` if you own a HuggingFace API token to speed up the model download and avoid resource limits. This is probably not required, but a failsafe in case we get rate-limited by HuggingFace.
   6. Select "Deploy Launchable" on the right. Your instance will start up, run its setup scripts, and start the vLLM server hosting your model. All of this will take some time, so you can already continue with the next steps.

   Now, the tasks for setting up manually:
   1. First visit https://brev.nvidia.com/environment/new, where you can select a machine with a certain number of selected GPUs. You don't need to select anything yet.
   2. Check out the [`qwen3.6-27b-vllm` directory](./qwen3.6-27b-vllm), which contains scripts to run on various GPUs. Pick a setup that sounds good and select the appropriate instance type:
      - L40S GPUs use a slightly less exact model and allow for fewer concurrent prompts, but are faster.
      - A100 GPUs are slower, but use the full-precision model and allow for more concurrent prompts.
      - Others GPUs use a slightly less exact model and allow for fewer concurrent prompts when you use the configurations that use only a single GPU. They are faster but generally less likely to be available.
      Remember to use at minimum as many GPUs as are in the filename (excluding those files that only use a single GPU).
   3. Start the instance and wait for it to boot up.
   4. Select "Open Notebook" to visit a JupyterLab on the instance.
   5. Select on the left, section select the big "+" button. In the big section on the right, under "Other", select "Text File". You can copy-and-paste, then save a setup script here. Alternatively, upload a file with the upload button on the left section, which looks like a wide, underlined up-arrow.
   6. Optionally, in the setup file, define a `VLLM_API_KEY` before the `docke run` command to secure your agent with an API key that you define.
   7. Optionally, in the setup file, define a `HF_TOKEN` before the `docke run` command if you own a HuggingFace API token to speed up the model download and avoid resource limits. This is probably not required, but a failsafe in case we get rate-limited by HuggingFace.
   8. In the big section on the right, under "Other", select "Terminal". You now have a remote terminal on the machine.
   9. Start the setup file on the terminal: `bash <filename>`
4. Expose the instance's vLLM port as a publicly accessible endpoint. Note the endpoint down for later. (If you used the launchable, you don't need to expose the endpoint, but you still need to set it to be publicly accessible and obtain the endpoint's URL.)
   1. After deploying the launchable, select "Go to instance page" on the right. If you deployed manually, you should already be on this page.
   2. In the "Secure Links" section:  
      If you used a launchable, find the endpoint starting with `https://vllm-`, select the "..." button on the right, then "Edit", then check the "Allow unauthenticated public access" box. If the endpoints only say "pending", you need to wait for the instance to finish booting up.

      If you didn't use a launchable, select "Share a Service" to expose the endpoint.  
      You need to enter "Port number" 8000, "Custom hostname" `vllm` and check the "Allow unauthenticated public access" box.
   3. Copy the link to the endpoint starting with `https://vllm-` and paste it somewhere you remember.

## Model

We'll be running the [FP8 version of Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) when we have explicit hardware support for its FP8 operations (e.g., on [L40S GPUs](https://www.nvidia.com/en-us/data-center/l40s/)). The FP8 version has much lower resource requirements while only having a minimal loss in quality compared to the [full-precision BF16 version](https://huggingface.co/Qwen/Qwen3.6-27B), which we'll be using on hardware without explicit FP8 support (e.g., [A100 GPUs](https://www.nvidia.com/en-us/data-center/a100/)).

Whichever version we use, we need quite a lot of computational power to run the model at full context length and at a useful speed for agentic tasks.

## Compute

We'll rent the computational power from [Brev](https://brev.nvidia.com/), an interface to various cloud providers, using the credit coupons distributed to you.

Because not all node types may be available at a given time point, we have some alternatives that all work. You can use the following GPUs, in this order of preference (mostly due to availability):

- ≥2×48GiB L40S GPUs
- ≥2×80GiB A100 GPUs
- ≥4×40GiB A100 GPUs
- 1×80GiB H100 GPU
- ≥2×80GiB H100 GPUs
- 1×141GiB H200 GPU
- ≥2×141GiB H200 GPUs
- 1×96GiB RTX Pro 6000 GPU
- ≥2×96GiB RTX Pro 6000 GPUs

If you get a message that the machine could not be provisioned, just try the next instance type on the list.

## Launchables

Launchables are pre-configured instance types with corresponding scripts and exposed endpoints.

- [≥2×48GiB L40S GPUs](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GJeHu2YSvuSBRNYnWYKztLvBL1)
- [≥2×80GiB A100 GPUs](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GP8XnhebvEq8YMzWgWvtK0GNh9)
- [≥4×40GiB A100 GPUs](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GP9lYkxUi6fOtZ5RAlZXinWkki)
- [1×80GiB H100 GPU](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GP9VGbzLcWiRUAuxc75ItzZhjy)
- [≥2×80GiB H100 GPUs](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GPAQhxgBMq8vBteHrAdaHP57Vn)
- [1×141GiB H200 GPU](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GPAgPA1WIYAHcohNd0LM6jqYZe)
- [≥2×141GiB H200 GPUs](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GPCIXGONnQFUtFrpjH2ktfZXP2)
- [1×96GiB RTX Pro 6000 GPU](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GPBDpmGQwKUiQ5VlxpmA19yKEH)
- [≥2×96GiB RTX Pro 6000 GPUs](https://brev.nvidia.com/launchable/deploy?launchableID=env-3GPBd7NNUFxVjhfJ3pdTXhvPVLW)

If you get a message that the machine could not be provisioned, just try the next instance type on the list.
