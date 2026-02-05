# Migration Plan: google.generativeai → google.genai

**Date**: 2026-02-05
**Status**: Planning Phase
**Priority**: High (google.generativeai support ended)

---

## Executive Summary

The `google.generativeai` package is deprecated and no longer receiving updates or bug fixes. Google has released a new unified SDK called `google.genai` that:

- Provides full feature parity with the deprecated package
- Supports both AI Studio (free tier) and Vertex AI (paid)
- Incorporates community feedback with improved API design
- Is actively maintained with ongoing support

**Deprecation Timeline:**
- `google-generativeai`: Deprecated as of August 31, 2025
- Vertex AI GenAI module: Deprecation on June 24, 2026
- SDK releases after June 24, 2026 won't include deprecated modules

---

## Impact Assessment

### Files Affected

1. **agent/core.py** - Main agent logic (PRIMARY)
   - Uses: `genai.configure()`, `GenerativeModel`, `FunctionDeclaration`, `Tool`, `start_chat()`, `send_message()`, `genai.protos`
   - Impact: High - Core functionality

2. **requirements.txt** - Dependencies
   - Current: `google-generativeai>=0.3.0`
   - New: `google-genai>=1.0.0`
   - Impact: Medium - Version constraint change

3. **tests/test_agent.py** - Unit tests
   - Uses: Mock patches for `agent.core.genai`
   - Impact: Medium - Mock structure changes

4. **docs/autoplot-agent-spec.md** - Documentation
   - Contains code examples with old API
   - Impact: Low - Documentation only

5. **spikes/mac_compatibility/** - Testing documentation
   - References `google-generativeai` package
   - Impact: Low - Historical record

### Current API Usage

```python
# Initialization
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Function declarations
from google.generativeai.types import FunctionDeclaration, Tool
function_declarations = []
for tool_schema in get_tool_schemas():
    fd = FunctionDeclaration(
        name=tool_schema["name"],
        description=tool_schema["description"],
        parameters=tool_schema["parameters"],
    )
    function_declarations.append(fd)
tools = Tool(function_declarations=function_declarations)

# Model initialization
self.model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=get_system_prompt(),
    tools=[tools],
)

# Chat session
self.chat = self.model.start_chat(history=[])
response = self.chat.send_message(user_message)

# Function responses
genai.protos.Part(
    function_response=genai.protos.FunctionResponse(
        name=tool_name,
        response={"result": result},
    )
)
```

---

## Migration Strategy

### Phase 1: Research & Planning ✅
- [x] Review deprecation notice
- [x] Study migration documentation
- [x] Analyze current codebase usage
- [x] Create migration plan document

### Phase 2: Environment Setup
- [ ] Install `google-genai` package alongside current package
- [ ] Test compatibility with Python 3.11
- [ ] Verify no conflicts with existing dependencies

### Phase 3: Code Migration
- [ ] Update `agent/core.py` with new API
- [ ] Update `requirements.txt`
- [ ] Update test mocks in `tests/test_agent.py`

### Phase 4: Testing & Validation
- [ ] Run unit tests (`pytest tests/`)
- [ ] Run macOS compatibility tests (`spikes/mac_compatibility/`)
- [ ] Manual integration testing with main.py
- [ ] Verify all 7 tools work correctly

### Phase 5: Documentation & Cleanup
- [ ] Update code comments in agent/core.py
- [ ] Update docs/autoplot-agent-spec.md examples
- [ ] Add migration notes to SESSION_SUMMARY.md
- [ ] Remove old package from requirements.txt

### Phase 6: Commit & Deploy
- [ ] Commit changes with detailed message
- [ ] Test on fresh environment
- [ ] Update README if necessary

---

## API Migration Reference

### 1. Initialization

**Before:**
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
```

**After:**
```python
from google import genai
client = genai.Client(api_key="YOUR_API_KEY")
# Or use environment variable GEMINI_API_KEY
client = genai.Client()
```

### 2. Model & Tools Configuration

**Before:**
```python
from google.generativeai.types import FunctionDeclaration, Tool

fd = FunctionDeclaration(
    name="tool_name",
    description="Tool description",
    parameters={"type": "object", "properties": {...}},
)
tools = Tool(function_declarations=[fd])

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="System prompt",
    tools=[tools],
)
```

**After:**
```python
from google import genai
from google.genai import types

client = genai.Client()

# Tools are passed directly as functions or function declarations
# System instructions are passed via config

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents="User message",
    config=types.GenerateContentConfig(
        tools=[tool_function],
        system_instruction="System prompt"
    )
)
```

### 3. Chat Session

**Before:**
```python
model = genai.GenerativeModel('gemini-2.5-flash')
chat = model.start_chat(history=[])
response = chat.send_message(user_message)
```

**After:**
```python
client = genai.Client()
chat = client.chats.create(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        system_instruction="System prompt",
        tools=[tool_function]
    )
)
response = chat.send_message(message=user_message)
```

### 4. Function Calling & Response

**Before:**
```python
# Access function calls
for part in response.candidates[0].content.parts:
    if hasattr(part, "function_call") and part.function_call.name:
        function_calls.append(part.function_call)

# Send function responses
function_responses.append(
    genai.protos.Part(
        function_response=genai.protos.FunctionResponse(
            name=tool_name,
            response={"result": result},
        )
    )
)
response = chat.send_message(function_responses)
```

**After:**
```python
# New API - need to verify exact structure from docs
# The new SDK handles function calling differently
# May use types.FunctionResponse or similar

# This section requires detailed testing to confirm exact API
```

---

## Code Changes Preview

### agent/core.py Changes

**Lines 5-6:**
```python
# BEFORE
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# AFTER
from google import genai
from google.genai import types
```

**Lines 28-48:**
```python
# BEFORE
genai.configure(api_key=GOOGLE_API_KEY)

function_declarations = []
for tool_schema in get_tool_schemas():
    fd = FunctionDeclaration(
        name=tool_schema["name"],
        description=tool_schema["description"],
        parameters=tool_schema["parameters"],
    )
    function_declarations.append(fd)

tools = Tool(function_declarations=function_declarations)

self.model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=get_system_prompt(),
    tools=[tools],
)

# AFTER
self.client = genai.Client(api_key=GOOGLE_API_KEY)

# Function declarations built from tool schemas
tool_functions = self._build_tool_functions()

self.config = types.GenerateContentConfig(
    system_instruction=get_system_prompt(),
    tools=tool_functions,
)

self.model_name = "gemini-2.5-flash"
```

**Lines 51:**
```python
# BEFORE
self.chat = self.model.start_chat(history=[])

# AFTER
self.chat = self.client.chats.create(
    model=self.model_name,
    config=self.config
)
```

**Lines 134:**
```python
# BEFORE
response = self.chat.send_message(user_message)

# AFTER
response = self.chat.send_message(message=user_message)
```

**Lines 176-186:**
```python
# BEFORE
function_responses.append(
    genai.protos.Part(
        function_response=genai.protos.FunctionResponse(
            name=tool_name,
            response={"result": result},
        )
    )
)
response = self.chat.send_message(function_responses)

# AFTER
# Need to verify exact function response structure in new API
# Likely uses types.FunctionResponse
function_response = types.FunctionResponse(
    name=tool_name,
    response=result
)
response = self.chat.send_message_with_function_response(function_response)
```

### requirements.txt Changes

```diff
- google-generativeai>=0.3.0
+ google-genai>=1.0.0
  jpype1==1.5.0
  python-dotenv>=1.0.0
  requests>=2.28.0
  pytest>=7.0.0
```

### tests/test_agent.py Changes

```python
# BEFORE
@pytest.fixture
def mock_genai(self):
    """Mock google.generativeai module."""
    with patch("agent.core.genai") as mock:
        yield mock

# AFTER
@pytest.fixture
def mock_genai(self):
    """Mock google.genai module."""
    with patch("agent.core.genai") as mock:
        # Mock Client, types, etc.
        mock_client = MagicMock()
        mock.Client.return_value = mock_client
        yield mock
```

---

## Testing Checklist

### Unit Tests
- [ ] `test_parse_relative_time` - No changes needed
- [ ] `test_format_tool_result` - No changes needed
- [ ] `test_search_datasets_tool` - Update mock patches
- [ ] `test_list_parameters_tool` - Update mock patches
- [ ] `test_ask_clarification_tool` - Update mock patches

### Integration Tests
- [ ] Agent initialization
- [ ] Tool schema registration
- [ ] Chat session creation
- [ ] Single message processing
- [ ] Function call detection
- [ ] Function call execution
- [ ] Function response handling
- [ ] Multi-turn conversation
- [ ] Chat history persistence
- [ ] Error handling

### Manual Testing
- [ ] `python main.py` - Start agent
- [ ] Search datasets command
- [ ] List parameters command
- [ ] Plot data command (requires Autoplot)
- [ ] Change time range command
- [ ] Export plot command
- [ ] Get plot info command
- [ ] Ask clarification command

### macOS Compatibility
- [ ] `pytest tests/ -v` - All 67 tests passing
- [ ] `spikes/mac_compatibility/run_all_tests.sh` - All tests passing
- [ ] Python 3.11 + Java 17 compatibility
- [ ] JPype integration still works
- [ ] No performance regression

---

## Risks & Mitigation

### Risk 1: API Breaking Changes
**Impact**: High
**Likelihood**: Medium
**Mitigation**:
- Keep old package installed during testing
- Test all features thoroughly before removing old package
- Document exact API differences discovered

### Risk 2: Function Calling Structure Changes
**Impact**: High
**Likelihood**: Medium
**Mitigation**:
- Carefully review function calling documentation
- Test with all 7 tools
- Create fallback mechanism if needed

### Risk 3: Chat Session Behavior Changes
**Impact**: Medium
**Likelihood**: Low
**Mitigation**:
- Test multi-turn conversations
- Verify history is preserved
- Test context retention

### Risk 4: Python 3.11 Compatibility
**Impact**: High
**Likelihood**: Low
**Mitigation**:
- Test in venv_py311 environment
- Run full test suite before committing
- Document any version constraints

### Risk 5: Performance Regression
**Impact**: Medium
**Likelihood**: Low
**Mitigation**:
- Measure test suite runtime before/after
- Monitor API response times
- Check for memory usage changes

---

## Rollback Plan

If migration fails:

1. **Immediate Rollback**
   ```bash
   git checkout .
   pip uninstall google-genai
   pip install google-generativeai>=0.3.0
   python -m pytest tests/ -v
   ```

2. **Partial Rollback**
   - Keep both packages installed
   - Create feature flag to switch between APIs
   - Test in parallel

3. **Document Issues**
   - Record exact error messages
   - Note which features failed
   - Report to google-genai GitHub issues

---

## Success Criteria

Migration is successful when:

- ✅ All 67 unit tests pass
- ✅ All macOS compatibility tests pass
- ✅ Manual testing shows all 7 tools work
- ✅ No performance regression
- ✅ No deprecation warnings
- ✅ Code is cleaner and more maintainable
- ✅ Documentation is updated

---

## Resources

- **Migration Guide**: https://ai.google.dev/gemini-api/docs/migrate
- **google-genai PyPI**: https://pypi.org/project/google-genai/
- **google-genai GitHub**: https://github.com/googleapis/python-genai
- **Deprecation Announcement**: https://github.com/google-gemini/deprecated-generative-ai-python
- **Medium Article**: https://medium.com/google-cloud/migrating-to-the-new-google-gen-ai-sdk-python-074d583c2350

---

## Timeline

| Phase | Duration | Start Date | Status |
|-------|----------|------------|--------|
| 1. Research & Planning | 1 hour | 2026-02-05 | ✅ Complete |
| 2. Environment Setup | 30 mins | TBD | ⏳ Pending |
| 3. Code Migration | 2 hours | TBD | ⏳ Pending |
| 4. Testing & Validation | 1 hour | TBD | ⏳ Pending |
| 5. Documentation | 30 mins | TBD | ⏳ Pending |
| 6. Commit & Deploy | 15 mins | TBD | ⏳ Pending |
| **Total** | **~5 hours** | | |

---

## Notes

- The old API will continue to work for existing projects, but won't receive updates
- This migration is preventive maintenance to ensure long-term support
- The new API is designed to be cleaner and more intuitive
- Function calling is a critical feature - requires careful testing
- Python 3.11 compatibility must be maintained (JPype requirement)

---

## Next Steps

1. **Approve Migration Plan** - Review this document
2. **Set Migration Window** - Choose time for migration
3. **Begin Phase 2** - Install google-genai package
4. **Test Iteratively** - Small changes, test frequently
5. **Document Discoveries** - Update this plan as we learn

---

**Last Updated**: 2026-02-05
**Created By**: Claude Code
**Review Status**: Awaiting user approval
