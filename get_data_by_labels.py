"""
数据提取工具
支持按标签和数量从MongoDB数据库中提取数据，并导出为多种格式
"""

import argparse
import json
import csv
import sys
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import getpass
from urllib.parse import quote_plus


class DataExtractor:
    """数据提取器"""
    
    def __init__(self, connection_string: str, database_name: str, random_seed: Optional[int] = None):
        """
        初始化数据提取器
        
        Args:
            connection_string: MongoDB连接字符串
            database_name: 数据库名称
            random_seed: 随机种子，用于保证结果可重现
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self.random_seed = random_seed
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
            print(f"✓ 设置随机种子: {random_seed}")
        
    def connect(self) -> bool:
        """连接到MongoDB数据库"""
        try:
            self.client = MongoClient(self.connection_string)
            # 测试连接
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            print(f"✓ 成功连接到数据库: {self.database_name}")
            return True
        except ConnectionFailure as e:
            print(f"✗ 数据库连接失败: {e}")
            return False
        except Exception as e:
            print(f"✗ 连接时发生错误: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            print("✓ 数据库连接已关闭")
    
    def get_available_labels(self) -> List[str]:
        """获取所有可用的标签"""
        try:
            pipeline = [
                {"$unwind": "$n_labels"},
                {"$group": {"_id": "$n_labels"}},
                {"$sort": {"_id": 1}}
            ]
            
            result = list(self.db.data_items.aggregate(pipeline))
            labels = [item["_id"] for item in result if item["_id"]]
            return labels
        except Exception as e:
            print(f"✗ 获取标签列表失败: {e}")
            return []
    
    def get_label_count(self, label: str) -> int:
        """获取指定标签的数据总数"""
        try:
            count = self.db.data_items.count_documents({"n_labels": label})
            return count
        except Exception as e:
            print(f"✗ 获取标签 '{label}' 数量失败: {e}")
            return 0
    
    def _extract_random_data(self, label: str, count: int, projection: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        随机提取指定标签的数据
        
        Args:
            label: 标签名称
            count: 需要提取的数量
            projection: 字段投影
            
        Returns:
            随机提取的数据列表
        """
        try:
            # 使用 MongoDB 的 $sample 聚合操作进行随机采样
            pipeline = [
                {"$match": {"n_labels": label}},
                {"$sample": {"size": count}},
                {"$project": projection}
            ]
            
            # 如果设置了随机种子，我们需要另一种方法来确保可重现性
            if self.random_seed is not None:
                # 获取所有匹配的文档ID，然后随机选择
                all_ids = list(self.db.data_items.find(
                    {"n_labels": label}, 
                    {"_id": 1}
                ))
                
                if len(all_ids) <= count:
                    # 如果总数不超过需求数量，直接返回所有数据
                    selected_ids = all_ids
                else:
                    # 随机选择指定数量的ID
                    selected_ids = random.sample(all_ids, count)
                
                # 根据选中的ID获取完整数据
                id_list = [doc["_id"] for doc in selected_ids]
                cursor = self.db.data_items.find(
                    {"_id": {"$in": id_list}},
                    projection
                )
                data_items = list(cursor)
            else:
                # 使用MongoDB的随机采样
                data_items = list(self.db.data_items.aggregate(pipeline))
            
            return data_items
            
        except Exception as e:
            print(f"  ✗ 随机提取数据失败: {e}")
            # 如果随机采样失败，退回到普通查询
            cursor = self.db.data_items.find(
                {"n_labels": label},
                projection
            ).limit(count)
            return list(cursor)
    
    def _process_data_item(self, item: Dict[str, Any], target_label: str) -> Dict[str, Any]:
        """
        处理单个数据项，特别是n_labels字段
        
        Args:
            item: 原始数据项
            target_label: 提取时使用的目标标签
            
        Returns:
            处理后的数据项
        """
        processed_item = item.copy()
        
        # 处理n_labels字段 - 只保留第一个元素
        if 'n_labels' in processed_item:
            n_labels = processed_item['n_labels']
            if isinstance(n_labels, list) and len(n_labels) > 0:
                processed_item['n_labels'] = n_labels[0]
            elif not isinstance(n_labels, list):
                # 如果不是列表，保持原值
                processed_item['n_labels'] = n_labels
            else:
                # 空列表的情况
                processed_item['n_labels'] = ""
        
        # 处理o_labels字段 - 转换为字符串表示
        if 'o_labels' in processed_item:
            o_labels = processed_item['o_labels']
            if isinstance(o_labels, list):
                processed_item['o_labels'] = ', '.join(str(label) for label in o_labels)
        
        # 添加提取时的标签信息
        processed_item['extracted_label'] = target_label
        
        return processed_item
    
    def extract_data_by_labels(self, label_requirements: Dict[str, int], 
                              fields: List[str], use_random: bool = False) -> List[Dict[str, Any]]:
        """
        按标签和数量提取数据
        
        Args:
            label_requirements: 标签和对应需要的数量 {label: count}
            fields: 需要提取的字段列表
            use_random: 是否使用随机采样
            
        Returns:
            所有提取的数据项列表
        """
        all_extracted_data = []
        
        # 构建字段投影
        projection = {field: 1 for field in fields}
        if "_id" not in fields:
            projection["_id"] = 0
        
        print(f"\n开始提取数据，共 {len(label_requirements)} 个标签...")
        if use_random:
            print(f"✓ 启用随机采样模式" + (f" (种子: {self.random_seed})" if self.random_seed else ""))
        
        for i, (label, required_count) in enumerate(label_requirements.items(), 1):
            print(f"\n[{i}/{len(label_requirements)}] 正在提取标签 '{label}' 的数据...")
            
            # 检查可用数量
            available_count = self.get_label_count(label)
            if available_count == 0:
                print(f"  ✗ 标签 '{label}' 没有可用数据")
                continue
            
            if available_count < required_count:
                print(f"  ⚠ 标签 '{label}' 可用数据 {available_count} 条，少于需求 {required_count} 条")
                actual_count = available_count
            else:
                actual_count = required_count
            
            try:
                if use_random:
                    # 使用随机采样
                    print(f"  📊 使用随机采样提取 {actual_count} 条数据...")
                    data_items = self._extract_random_data(label, actual_count, projection)
                else:
                    # 使用普通查询
                    cursor = self.db.data_items.find(
                        {"n_labels": label},
                        projection
                    ).limit(actual_count)
                    data_items = list(cursor)
                
                if data_items:
                    # 处理数据项
                    processed_items = [self._process_data_item(item, label) for item in data_items]
                    all_extracted_data.extend(processed_items)
                    print(f"  ✓ 成功提取 {len(data_items)} 条数据")
                else:
                    print(f"  ⚠ 标签 '{label}' 查询结果为空")
                
            except Exception as e:
                print(f"  ✗ 提取标签 '{label}' 数据失败: {e}")
                import traceback
                print(f"  详细错误: {traceback.format_exc()}")
                continue
        
        return all_extracted_data
    
    def save_to_excel(self, data: List[Dict[str, Any]], output_path: str):
        """保存数据到Excel文件（单个工作表）"""
        try:
            if data:
                df = pd.DataFrame(data)
                
                # 重新排列列的顺序，把extracted_label放在前面
                columns = list(df.columns)
                if 'extracted_label' in columns:
                    columns.remove('extracted_label')
                    columns.insert(0, 'extracted_label')
                    df = df[columns]
                
                df.to_excel(output_path, sheet_name='extracted_data', index=False)
                print(f"✓ 数据已保存到Excel文件: {output_path}")
                print(f"  总计保存 {len(data)} 条数据")
                
                # 显示各标签的数据统计
                if 'extracted_label' in df.columns:
                    label_counts = df['extracted_label'].value_counts()
                    print(f"  各标签数据统计:")
                    for label, count in label_counts.items():
                        print(f"    {label}: {count} 条")
            else:
                print("⚠ 没有数据可保存")
        except Exception as e:
            print(f"✗ 保存Excel文件失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
    
    def save_to_csv(self, data: List[Dict[str, Any]], output_path: str):
        """保存数据到CSV文件"""
        try:
            if data:
                df = pd.DataFrame(data)
                
                # 重新排列列的顺序，把extracted_label放在前面
                columns = list(df.columns)
                if 'extracted_label' in columns:
                    columns.remove('extracted_label')
                    columns.insert(0, 'extracted_label')
                    df = df[columns]
                
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"✓ 数据已保存到CSV文件: {output_path}")
                print(f"  总计保存 {len(data)} 条数据")
            else:
                print("⚠ 没有数据可保存")
        except Exception as e:
            print(f"✗ 保存CSV文件失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
    
    def save_to_jsonl(self, data: List[Dict[str, Any]], output_path: str):
        """保存数据到JSONL文件"""
        try:
            if data:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"✓ 数据已保存到JSONL文件: {output_path}")
                print(f"  总计保存 {len(data)} 条数据")
            else:
                print("⚠ 没有数据可保存")
        except Exception as e:
            print(f"✗ 保存JSONL文件失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")


def parse_label_requirements(label_spec: str) -> Dict[str, int]:
    """
    解析标签需求规格
    格式: "label1:count1,label2:count2,..."
    例如: "violence:1000,spam:500,normal:2000"
    """
    requirements = {}
    try:
        pairs = label_spec.split(',')
        for pair in pairs:
            if ':' in pair:
                label, count_str = pair.strip().split(':', 1)
                count = int(count_str.strip())
                if count > 0:
                    requirements[label.strip()] = count
                else:
                    print(f"⚠ 忽略无效数量: {pair}")
            else:
                print(f"⚠ 忽略无效格式: {pair}")
    except ValueError as e:
        print(f"✗ 解析标签需求失败: {e}")
        return {}
    
    return requirements


def main():
    parser = argparse.ArgumentParser(
        description="从MongoDB数据库中按标签提取训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 普通提取
  python get_data_by_labels.py -l "violence:1000,spam:500,normal:2000" -f text_cn,o_labels,n_labels -o output.xlsx
  
  # 随机提取（可重现）
  python get_data_by_labels.py -l "正常:5000,暴恐:1000" -f text_cn,n_labels --random --seed 42 -o random_output.xlsx
  python get_data_by_labels.py -u admin --auth-database scanadata -l "正常:12000,暴恐:3000,歧视:3000,色情低俗:3000,违法违规:3000" -f text_cn,n_labels --random --seed 42 -o r-42-4-3000.xlsx
  # 训练数据集
  python get_data_by_labels.py -u admin --auth-database scanadata -l "正常:15000,政治安全:6000,暴恐:6000,歧视:6000,色情低俗:6000,违法违规:6000" -f text_cn,n_labels --random --seed 123 -o r123-5-6000.xlsx
  # 独立验证数据集
  python get_data_by_labels.py -u admin --auth-database scanadata -l "正常:3000,政治安全:3000,暴恐:3000,歧视:3000,色情低俗:3000,违法违规:3000" -f text_cn,n_labels --random --seed 456 -o val-r456-6-3000.xlsx
  
  # 随机提取（每次不同）
  python get_data_by_labels.py -l "正常:5000,暴恐:1000" -f text_cn,n_labels --random -o random_output.xlsx
  
  # 查看所有可用标签
  python get_data_by_labels.py --list-labels -u admin --auth-database scanadata
        """
    )
    
    parser.add_argument('--host', default='10.8.24.135',
                       help='MongoDB主机地址 (默认: 10.8.24.135)')
    parser.add_argument('--port', type=int, default=27217,
                       help='MongoDB端口 (默认: 27217)')
    parser.add_argument('--database', '-d', default='scanadata',
                       help='数据库名称 (默认: scanadata)')
    parser.add_argument('--username', '-u',
                       help='MongoDB用户名')
    parser.add_argument('--auth-database', default='admin',
                       help='认证数据库 (默认: admin)')
    
    parser.add_argument('--labels', '-l',
                       help='标签和数量规格，格式: "label1:count1,label2:count2"')
    parser.add_argument('--fields', '-f', 
                       default='text_cn,o_labels,n_labels,source_id',
                       help='要提取的字段，用逗号分隔 (默认: text_cn,o_labels,n_labels,source_id)')
    parser.add_argument('--format', choices=['excel', 'csv', 'jsonl'], 
                       default='excel',
                       help='输出格式 (默认: excel)')
    parser.add_argument('--output', '-o',
                       help='输出文件路径')
    
    parser.add_argument('--random', action='store_true',
                       help='启用随机采样模式')
    parser.add_argument('--seed', type=int,
                       help='随机种子，用于保证结果可重现')
    
    parser.add_argument('--list-labels', action='store_true',
                       help='列出所有可用标签并退出')
    
    args = parser.parse_args()
    
    # 构建连接字符串
    if args.username:
        password = getpass.getpass(f"请输入用户 {args.username} 的密码: ")
        encoded_username = quote_plus(args.username)
        encoded_password = quote_plus(password)
        connection_string = f"mongodb://{encoded_username}:{encoded_password}@{args.host}:{args.port}/{args.auth_database}"
    else:
        connection_string = f"mongodb://{args.host}:{args.port}/"
    
    # 创建提取器（包含随机种子）
    extractor = DataExtractor(connection_string, args.database, args.seed)
    
    try:
        # 连接数据库
        if not extractor.connect():
            sys.exit(1)
        
        # 如果只是查看标签
        if args.list_labels:
            print("\n可用标签列表:")
            labels = extractor.get_available_labels()
            if labels:
                for label in labels:
                    count = extractor.get_label_count(label)
                    print(f"  {label}: {count} 条数据")
            else:
                print("  (没有找到标签)")
            return
        
        # 检查必需参数
        if not args.labels:
            print("✗ 请指定要提取的标签和数量 (-l 参数)")
            sys.exit(1)
        
        # 解析标签需求
        label_requirements = parse_label_requirements(args.labels)
        if not label_requirements:
            print("✗ 没有有效的标签需求")
            sys.exit(1)
        
        print(f"\n提取计划:")
        total_required = 0
        for label, count in label_requirements.items():
            available = extractor.get_label_count(label)
            print(f"  {label}: 需要 {count} 条，可用 {available} 条")
            total_required += min(count, available)
        
        print(f"\n预计提取总数: {total_required} 条")
        
        # 解析字段列表
        fields = [field.strip() for field in args.fields.split(',') if field.strip()]
        print(f"提取字段: {', '.join(fields)}")
        
        # 生成输出文件名
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extensions = {'excel': 'xlsx', 'csv': 'csv', 'jsonl': 'jsonl'}
            random_suffix = "_random" if args.random else ""
            seed_suffix = f"_seed{args.seed}" if args.seed else ""
            args.output = f"extracted_data{random_suffix}{seed_suffix}_{timestamp}.{extensions[args.format]}"
        
        # 提取数据
        print(f"\n开始数据提取...")
        extracted_data = extractor.extract_data_by_labels(label_requirements, fields, args.random)
        
        if not extracted_data:
            print("✗ 没有提取到任何数据")
            sys.exit(1)
        
        # 保存数据
        print(f"\n保存数据到 {args.format.upper()} 格式...")
        if args.format == 'excel':
            extractor.save_to_excel(extracted_data, args.output)
        elif args.format == 'csv':
            extractor.save_to_csv(extracted_data, args.output)
        elif args.format == 'jsonl':
            extractor.save_to_jsonl(extracted_data, args.output)
        
        # 输出最终统计信息
        print(f"\n提取完成统计:")
        if extracted_data:
            # 按标签统计
            label_stats = {}
            for item in extracted_data:
                label = item.get('extracted_label', 'Unknown')
                label_stats[label] = label_stats.get(label, 0) + 1
            
            for label, count in label_stats.items():
                print(f"  {label}: {count} 条")
            print(f"总计: {len(extracted_data)} 条数据")
            
            # 如果使用了随机模式，给出提示
            if args.random:
                if args.seed:
                    print(f"\n💡 使用了随机种子 {args.seed}，相同种子可重现此结果")
                else:
                    print(f"\n💡 使用了随机采样，每次运行结果可能不同")
        
    finally:
        extractor.close()


if __name__ == "__main__":
    main()