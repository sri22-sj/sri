import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME
from agents.requirement_agent import requirement_prompt
from agents.architecture_agent import architecture_prompt
from agents.coding_agent import coding_prompt
from agents.debugging_agent import debugging_prompt

from tools.project_generator import create_project
from tools.code_executor import run_python_file


class ProjectPilotAgent:

    def __init__(self):

        print("Loading ProjectPilot AI...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )

        print("ProjectPilot ready.")

    def generate(self, prompt):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are ProjectPilot AI, an autonomous "
                    "software development agent."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=False
            )

        generated = output[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

    def analyze_requirements(self, idea):

        prompt = requirement_prompt(idea)

        return self.generate(prompt)

    def design_architecture(self, requirements):

        prompt = architecture_prompt(requirements)

        return self.generate(prompt)

    def generate_code(self, idea, architecture):

        prompt = coding_prompt(
            idea,
            architecture
        )

        return self.generate(prompt)

    def parse_files(self, response):

        pattern = r"FILE:\s*(.*?)\s*```(?:python|text|markdown)?\s*(.*?)```"

        matches = re.findall(
            pattern,
            response,
            re.DOTALL
        )

        files = []

        for path, content in matches:

            files.append({
                "path": path.strip(),
                "content": content.strip()
            })

        return files

    def run(self, project_idea):

        print("\n[1] Analyzing requirements...")

        requirements = self.analyze_requirements(
            project_idea
        )

        print("\n[2] Designing architecture...")

        architecture = self.design_architecture(
            requirements
        )

        print("\n[3] Generating code...")

        code_response = self.generate_code(
            project_idea,
            architecture
        )

        files = self.parse_files(
            code_response
        )

        if not files:

            return {
                "success": False,
                "message": "No files were generated."
            }

        project_name = "generated_project"

        project_path = create_project(
            project_name,
            files
        )

        print(
            f"\nProject created at: {project_path}"
        )

        return {
            "success": True,
            "requirements": requirements,
            "architecture": architecture,
            "files": files,
            "project_path": project_path
        }
