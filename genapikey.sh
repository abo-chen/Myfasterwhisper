#!/bin/bash
# API Key Generator / API Key 生成器
# Usage: ./genapikey.sh [count] [options]
# 用法: ./genapikey.sh [数量] [选项]

set -e

# Default values / 默认值
COUNT=1
PREFIX="sk"
OUTPUT_FILE=""
SHOW_HELP=false

# Colors for output / 输出颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage / 打印帮助
print_help() {
    cat << EOF
API Key Generator / API Key 生成器

Usage / 用法:
  $0 [OPTIONS] [COUNT]

Options / 选项:
  -c, --count NUM      Number of keys to generate (default: 1)
                      生成密钥数量（默认：1）
  -p, --prefix PREFIX  Key prefix, e.g., 'sk', 'pk' (default: sk)
                      密钥前缀，如 'sk', 'pk'（默认：sk）
  -o, --output FILE    Save keys to file
                      保存密钥到文件
  -h, --help           Show this help message
                      显示帮助信息

Examples / 示例:
  $0                           # Generate 1 key / 生成 1 个密钥
  $0 -c 5                      # Generate 5 keys / 生成 5 个密钥
  $0 -c 10 -o keys.txt         # Generate 10 keys and save to file
                                生成 10 个密钥并保存到文件
  $0 -p myapp -c 3             # Generate 3 keys with 'myapp' prefix
                                生成 3 个带 'myapp' 前缀的密钥

EOF
}

# Parse arguments / 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--count)
            COUNT="$2"
            shift 2
            ;;
        -p|--prefix)
            PREFIX="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use -h for help / 使用 -h 查看帮助"
            exit 1
            ;;
        *)
            COUNT="$1"
            shift
            ;;
    esac
done

# Show help if requested / 如果请求则显示帮助
if [ "$SHOW_HELP" = true ]; then
    print_help
    exit 0
fi

# Validate count / 验证数量
if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [ "$COUNT" -lt 1 ]; then
    echo -e "${RED}Error: Count must be a positive number${NC}"
    exit 1
fi

# Generate keys using Python secrets / 使用 Python secrets 生成密钥
generate_keys() {
    python3 << PYTHON_SCRIPT
import secrets

prefix = "$PREFIX"
count = $COUNT

for _ in range(count):
    # Generate 32-byte random key / 生成 32 字节随机密钥
    raw = secrets.token_urlsafe(32)

    # Format with prefix / 带前缀格式化
    key = f"{prefix}-{raw}"

    print(key)
PYTHON_SCRIPT
}

# Main logic / 主逻辑
echo -e "${GREEN}Generating $COUNT API key(s)...${NC}"
echo ""

# Generate keys to array / 生成密钥到数组
mapfile -t KEYS < <(generate_keys)

# Display keys / 显示密钥
echo -e "${YELLOW}Generated API Keys:${NC}"
echo "-------------------"
for i in "${!KEYS[@]}"; do
    echo "$((i+1)). ${KEYS[$i]}"
done
echo "-------------------"
echo ""

# Save to file if specified / 如果指定则保存到文件
if [ -n "$OUTPUT_FILE" ]; then
    echo "${KEYS[@]}" | tr ' ' '\n' > "$OUTPUT_FILE"
    echo -e "${GREEN}Keys saved to: $OUTPUT_FILE${NC}"
fi

# Summary / 摘要
echo -e "${GREEN}✓ Successfully generated $COUNT API key(s)${NC}"
echo -e "  Prefix: ${YELLOW}$PREFIX${NC}"
if [ -n "$OUTPUT_FILE" ]; then
    echo -e "  Output: ${YELLOW}$OUTPUT_FILE${NC}"
fi
echo ""
echo -e "${YELLOW}⚠ Keep these keys secure and don't share them!${NC}"
echo -e "${YELLOW}⚠ 请妥善保管这些密钥，不要泄露！${NC}"
