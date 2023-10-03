def get_latency_s(latency):
    return 1 - (latency - 1.) / (200. - 1.)

def get_memory_s(mem):
    return 1 - (mem - 5.) / (64. - 5.)

def get_memory_old_s(mem):
    return 1 - (mem - 5.) / (256. - 5.)

def last_year(fbeta, latency, mem):
    latency_score = (1 - (latency - 1)/(200 - 1))
    mem_score = (1 - (mem - 5)/(256 - 5))
    return 100 * fbeta + 20 * latency_score + 20 * mem_score

print(get_latency_s(5.6)*20, get_memory_s(15.0)*20)
print(get_latency_s(5.6)*20+get_memory_s(17.33)*20+96.897775727448415)
print(get_latency_s(5.6)*20+get_memory_s(15.0)*20+96.897775727448415)
print(get_latency_s(5.6)*20+get_memory_s(14.2)*20+96.897775727448415)
print(get_latency_s(5.6)*20+get_memory_s(13.0)*20+96.897775727448415)
print(f"Un quant {get_latency_s(5.0)*20+get_memory_s(14.452)*20+97}")
print(f"float16 quant {get_latency_s(5.0)*20+get_memory_s(12.538)*20+96.73}")
print(f"float16 mem{get_memory_s(12.538)*20}, int8 mem{get_memory_s(11.18)*20}")
print(f"MIT baseline {get_latency_s(0.8)*20+get_memory_s(11.18)*20+94}")