local json = require("json")
local http = require("socket.http")
local url = require("socket.url")
local ltn12 = require("ltn12")


local headers = {
  Authorization = 'Bearer **'
}

local base_url = "http://127.0.0.1:5000"


local function fetch_text(url, op)
  local response_body = {}

  local request_op = {
    url = url,
    sink = ltn12.sink.table(response_body)
  }

  if op then
    for k, v in pairs(op) do
      if k ~= 'url' then
        request_op[k] = v
      end
    end
  end

  local _, code = http.request(request_op)

  -- 将response_body数组转换为字符串
  local response_str = table.concat(response_body)

  return code, response_str
end

local translator = {}

---@param env Env
function translator.init(env)
  env.memory = Memory(env.engine, env.engine.schema)
  env.notifier = env.engine.context.commit_notifier:connect(function(ctx)
    local commit = ctx.commit_history:back()
    if commit then
      fetch_text(base_url .. "/commit?text=" .. url.escape(commit.text), {
        headers = headers
      })
    end
  end)
end

function translator.fini(env)
  env.notifier:disconnect()
  env.memory:disconnect()
  env.memory = nil
  collectgarbage()
end

---@param input string
---@param seg Segment
---@param env Env
function translator.func(input, seg, env)
  local qp = input
  local code, reply = fetch_text(base_url .. "/candidates?keys=" .. url.escape(qp), {
    headers = headers
  })
  local _, j = pcall(json.decode, reply)
  if code == 200 and _ then
    for i, v in ipairs(j.candidates) do
      local word = string.gsub(v['word'], "'", " ")
      local c = Candidate("normal", seg.start, seg._end, word, "")
      c.quality = 2
      c.preedit = v["preedit"]
      yield(c)
    end
  end
end

return { translator = translator }
