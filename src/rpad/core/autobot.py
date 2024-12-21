import subprocess
from io import StringIO

import pandas as pd

AUTOBOT_URL = "autobot.vision.cs.cmu.edu"

NODE_INFO = {
    "autobot-0-9": {"RTX_2080_Ti": 4},
    "autobot-0-11": {"RTX_2080_Ti": 4},
    "autobot-0-13": {"RTX_2080_Ti": 4},
    "autobot-0-15": {"RTX_2080_Ti": 4},
    "autobot-0-17": {"RTX_2080_Ti": 3},
    "autobot-0-19": {"RTX_2080_Ti": 4},
    "autobot-0-21": {"RTX_2080_Ti": 4},
    "autobot-0-23": {"RTX_2080_Ti": 4},
    "autobot-0-25": {"RTX_3090": 8},
    "autobot-0-29": {"RTX_3090": 8},
    "autobot-0-33": {"RTX_3090": 8},
    "autobot-0-37": {"RTX_3090": 8},
    "autobot-1-1": {"RTX_2080_Ti": 10},
    "autobot-1-1": {"RTX_3080_Ti": 8},
    "autobot-1-14": {"RTX_3080_Ti": 8},
    "autobot-1-18": {"RTX_A6000": 8},
}

# get a list of all the pids for each.
USER_LIST_CMD = r"nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//' | xargs -r ps --no-headers -up"

# Get the PID and serial number.
PID_USAGE_CMD = r"nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv"

# Get the GPU current usage info.
USAGE_CMD = (
    r"nvidia-smi --query-gpu=index,name,memory.used,memory.total,gpu_uuid --format=csv"
)


def parse_usage_cmd(response):
    df = pd.read_csv(StringIO(response), sep=", ", engine="python")
    # Weirdly, there's a space in uuid.
    df = df.rename(columns={"uuid": "gpu_uuid"})
    return df


def parse_pid_usage_cmd(response):
    return pd.read_csv(StringIO(response), sep=", ", engine="python")


def parse_user_list_cmd(response):
    lines = response.split("\n")
    new_resp = []
    for line in lines:
        if line == "":
            continue
        # .split() will remove variable length spaces.
        new_resp.append(",".join(line.split()[:2]))
    new_resp_str = "\n".join(new_resp)
    return pd.read_csv(StringIO(new_resp_str), sep=",", names=["username", "pid"])


def execute_command_chain_on_node(node_name, commands, username=None, local=False):
    split_token = "END_OF_COMMAND_OUTPUT"
    new_commands = []
    for command in commands:
        new_commands.append(command)
        new_commands.append(f"echo {split_token}")
    new_commands = new_commands[:-1]
    joined_cmd = " && ".join(new_commands)
    outputs = execute_command_on_node(
        node_name, joined_cmd, username=username, local=local
    )

    return outputs.split(split_token)


def execute_command_on_node(node_name, command, username=None, local=False):
    # If we're already logged into, say
    if local:
        raise ValueError("Not supported yet")
    else:
        if username is None:
            raise ValueError("username must be provided if local=True")
        cmd = ["ssh", f"{username}@{AUTOBOT_URL}", f'ssh {node_name} "{command}"']
        output = subprocess.check_output(cmd, text=True)
        return output


if __name__ == "__main__":
    # raw_result = execute_command_on_node(node_name="autobot-0-29", command=USAGE_CMD, username="baeisner", local=False)
    # print(raw_result)
    for node in NODE_INFO.keys():
        print(f"--------Node {node}----------")

        try:

            raw_results = execute_command_chain_on_node(
                # node_name="autobot-0-29",
                node_name=node,
                commands=[
                    USAGE_CMD,
                    PID_USAGE_CMD,
                    USER_LIST_CMD,
                ],
                username="baeisner",
            )

            # breakpoint()
            usage_df = parse_usage_cmd(raw_results[0])

            pid_usage_df = parse_pid_usage_cmd(raw_results[1])
            user_map_df = parse_user_list_cmd(raw_results[2])

            # Merg
            process_df = pd.merge(pid_usage_df, user_map_df, on="pid")
            process_df = pd.merge(usage_df, process_df, on="gpu_uuid")

            usage_df["processes"] = usage_df["gpu_uuid"].apply(
                lambda gpu_uuid: "\n".join(
                    process_df[process_df["gpu_uuid"] == gpu_uuid].apply(
                        lambda row: f"{row['username']} ({row['pid']}) [{row['memory.used [MiB]']}]",
                        axis=1,
                    )
                )
            )

            usage_df = usage_df.drop(columns=["gpu_uuid"])
            print(usage_df)
        except:
            print("\t FAILED")
