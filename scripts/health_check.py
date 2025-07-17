import asyncio
import aiohttp
import sys
import logging

async def check_health():
    """Comprehensive health check"""
    
    checks = {
        "streamlit_app": False,
        "vector_store": False,
        "llm_connection": False
    }
    
    # Check Streamlit app
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8501/_stcore/health", timeout=5) as response:
                checks["streamlit_app"] = response.status == 200
    except Exception as e:
        logging.error(f"Streamlit health check failed: {e}")
    
    # Additional checks would go here...
    
    # Report results
    all_healthy = all(checks.values())
    
    if all_healthy:
        print("✅ All systems healthy")
        sys.exit(0)
    else:
        print("❌ Some systems unhealthy:")
        for service, status in checks.items():
            status_emoji = "✅" if status else "❌"
            print(f"  {status_emoji} {service}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(check_health())