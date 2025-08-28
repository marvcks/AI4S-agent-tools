#!/usr/bin/env python3
"""
ASKCOS MCP Server for computer-aided synthesis planning.
Provides retrosynthesis planning, synthesis route analysis, and reaction prediction tools.
"""
import argparse
import os
import json
import logging
from typing import Dict, List, Optional, Any

import requests
from mcp.server.fastmcp import FastMCP

# 从环境变量获取 ASKCOS API URL，如果未设置则使用默认值
ASKCOS_API_URL = os.environ.get("ASKCOS_API_URL", "http://localhost:8080/askcos/api")

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="ASKCOS MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = parse_args()
mcp = FastMCP("ASKCOS", host=args.host, port=args.port)

logger.info(f"ASKCOS API URL: {ASKCOS_API_URL}")

# Define ASKCOS tools at module level
@mcp.tool()
def retrosynthesis_planning(
    smiles: str, 
    max_steps: int = 5, 
    top_n: int = 1,
    template_set: str = "reaxys",
    max_branching: int = 25
) -> Dict[str, Any]:
    """
    执行逆合成规划，为给定的目标分子生成合成路线。
    
    Args:
        smiles: 目标分子的SMILES字符串
        max_steps: 最大规划步骤数 (默认: 5)
        top_n: 返回的最佳路线数量 (默认: 1)
        template_set: 使用的模板集 (默认: "reaxys")
        max_branching: 最大分支数 (默认: 25)
        
    Returns:
        包含逆合成规划结果的字典
    """
    logger.info(f"执行逆合成规划: {smiles}")
    
    try:
        payload = {
            "smiles": smiles,
            "max_steps": max_steps,
            "top_n": top_n,
            "template_set": template_set,
            "max_branching": max_branching
        }
        
        response = requests.post(
            f"{ASKCOS_API_URL}/retrosynthesis",
            json=payload,
            timeout=300  # 5分钟超时
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"逆合成规划成功完成: {smiles}")
        
        return {
            "success": True,
            "smiles": smiles,
            "routes": result.get("routes", []),
            "num_routes": len(result.get("routes", [])),
            "parameters": payload
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"逆合成规划请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "smiles": smiles
        }
    except Exception as e:
        error_msg = f"逆合成规划过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "smiles": smiles
        }

@mcp.tool()
def synthesis_route_analysis(route_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析给定的合成路线，评估其可行性和复杂度。
    
    Args:
        route_json: 合成路线的JSON表示
        
    Returns:
        包含路线分析结果的字典
    """
    logger.info("执行合成路线分析")
    
    try:
        response = requests.post(
            f"{ASKCOS_API_URL}/route_analysis",
            json=route_json,
            timeout=120  # 2分钟超时
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info("合成路线分析成功完成")
        
        return {
            "success": True,
            "analysis": result,
            "route_complexity": result.get("complexity_score", "未知"),
            "feasibility": result.get("feasibility_score", "未知")
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"合成路线分析请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"合成路线分析过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

@mcp.tool()
def reaction_prediction(
    reactants_smiles: str, 
    reagents_smiles: Optional[str] = None, 
    solvent_smiles: Optional[str] = None,
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    预测给定反应物和试剂的反应产物。
    
    Args:
        reactants_smiles: 反应物的SMILES字符串
        reagents_smiles: 试剂的SMILES字符串 (可选)
        solvent_smiles: 溶剂的SMILES字符串 (可选)
        temperature: 反应温度 (可选)
        
    Returns:
        包含反应预测结果的字典
    """
    logger.info(f"执行反应预测: {reactants_smiles}")
    
    try:
        payload = {"reactants": reactants_smiles}
        if reagents_smiles:
            payload["reagents"] = reagents_smiles
        if solvent_smiles:
            payload["solvent"] = solvent_smiles
        if temperature is not None:
            payload["temperature"] = temperature
        
        response = requests.post(
            f"{ASKCOS_API_URL}/reaction_prediction",
            json=payload,
            timeout=120  # 2分钟超时
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"反应预测成功完成: {reactants_smiles}")
        
        return {
            "success": True,
            "reactants": reactants_smiles,
            "products": result.get("products", []),
            "confidence": result.get("confidence", "未知"),
            "conditions": {
                "reagents": reagents_smiles,
                "solvent": solvent_smiles,
                "temperature": temperature
            }
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"反应预测请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "reactants": reactants_smiles
        }
    except Exception as e:
        error_msg = f"反应预测过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "reactants": reactants_smiles
        }


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)