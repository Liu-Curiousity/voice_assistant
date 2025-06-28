def light_on():
    print("==>打开灯光<==")
    return "success"

def light_off():
    print("==>关闭灯光<==")
    return "success"

def light_brighter():
    print("==>调高灯光亮度<==")
    return "success"

def light_dimmer():
    print("==>调低灯光亮度<==")
    return "success"

tools_map = {"light_on": light_on,
            "light_off": light_off}
