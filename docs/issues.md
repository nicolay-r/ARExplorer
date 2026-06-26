KNOWN ISSUES:
- problem with TEXT parameter when perform classfiication.
- problem with large output from tools. 
    It should be mitigated to use of SQLITE (or the one that is better for JSON) storage for the output. 
    The output of each tool must be presented in individual table.
    The results shared as path to the corresponding table.
- Pydantic could not validate output in several cases:
    ```
        File "/home/nicolay/proj/ARExplorer/.venv/lib/python3.10/site-packages/pydantic/main.py", line 732, in model_validate
            return cls.__pydantic_validator__.validate_python(
        pydantic_core._pydantic_core.ValidationError: 1 validation error for AgentResponse
    ```