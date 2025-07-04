import motor.motor_asyncio
import asyncio

async def test_connection():
    client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
    try:
        # Send a ping to confirm a successful connection
        await client.admin.command('ping')
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())