#!/usr/bin/env bash
# Cross-shell compatible environment setup script
# Works with both bash and zsh

# Set environment variables based on user
if [ "$USER" = "iyngkarrankumar" ]; then
    echo "Working on local machine, USER: $USER"
    echo "Host: $HOSTNAME"
    export node_type="local"

    # Source local environment variables
    set -a
    source env/.env.local
    set +a

elif [ "$USER" = "s2517451" ]; then
    echo "Working on Edinburgh cluster machine, USER: $USER"
    echo "Host: $HOSTNAME"

    # Check if on Eddie cluster
    if [[ "$HOSTNAME" == *"ecdf.ed.ac.uk"* ]]; then
        if [[ "$HOSTNAME" == *"login"* ]]; then
            echo "On eddie head node"
            export node_type="eddie_head"
            set -a
            source env/.env.eddie_head
            set +a
        elif [[ "$HOSTNAME" == *"node"* ]]; then
            echo "On eddie compute node"
            export node_type="eddie_compute"
            set -a
            source env/.env.eddie_compute
            set +a
        else
            echo "On eddie cluster but unknown node type"
            export node_type="eddie_unknown"
        fi
    # Check if on MLP cluster
    elif [ -n "$SLURM_JOB_ID" ] || [ -n "$SLURMD_NODENAME" ]; then
        echo "On mlp compute node"
        export node_type="mlp_compute"
        set -a
        source env/.env.mlp_compute
        set +a
    else
        echo "On mlp head node"
        export node_type="mlp_head"
        set -a
        source env/.env.mlp_head
        set +a
    fi

else
    echo "Working on vast.ai machine, USER: $USER"
    echo "Host: $HOSTNAME"
    export node_type="vastai"
    set -a
    source env/.env.vastai
    set +a
fi

echo "node_type: $node_type"

eval "$(mamba shell hook --shell bash)"

mamba activate ml_env