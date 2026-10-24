import asyncio
import struct
import csv
import os
import sys
import time
from functools import reduce
from bleak import BleakClient, BleakScanner

#read me: this is the movesense python client available on their github but for two sensors

# Sensor last few serial numbers, change for different sensors!!!!
SENSOR_MAP = {
    "1033": "chest",
    "231":  "wrist",
}

OUTPUT_FOLDER = "data_live"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

RECORDER_LOG = os.path.join(OUTPUT_FOLDER, "recorder.log")

# Tee stdout → console + log file for streamlit to read recording status
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

_log_file = open(RECORDER_LOG, 'w', buffering=1)   # line-buffered
sys.stdout = _Tee(sys.__stdout__, _log_file)

# GATT UUIDs
WRITE_UUID  = "34800001-7185-4d5d-b431-630e7050e8f0" #these may be different for different sensors
NOTIFY_UUID = "34800002-7185-4d5d-b431-630e7050e8f0"

#command codes
#available from movesense documentation
REF_UTCTIME      = 10   # PUT_UTCTIME
REF_TIME_DETAIL  = 11   # GET /Time/Detailed
REF_STREAM       = 99   # SUBSCRIBE IMU9



class DataView:
    def __init__(self, array, bytes_per_element=1):
        self.array = array
        self.bytes_per_element = 1

    def __get_binary(self, start_index, byte_count, signed=False):
        integers = [self.array[start_index + x] for x in range(byte_count)]
        _bytes   = [i.to_bytes(self.bytes_per_element, byteorder='little', signed=signed)
                    for i in integers]
        return reduce(lambda a, b: a + b, _bytes)

    def get_uint_8(self,  start_index): return int.from_bytes(self.__get_binary(start_index, 1), byteorder='little')
    def get_int_64(self,  start_index): return struct.unpack('<q', self.__get_binary(start_index, 8))[0]
    def get_uint_32(self, start_index): return struct.unpack('<I', self.__get_binary(start_index, 4))[0]
    def get_float_32(self,start_index): return struct.unpack('<f', self.__get_binary(start_index, 4))[0]




async def run_queue_consumer(queue):
    active_files = {}
    print(f"Storage: Saving data to '{OUTPUT_FOLDER}/'...")

    while True:
        data = await queue.get()
        if data is None:
            break

        row        = data.split(',')
        serial_key = row[0]

        label = "unknown"
        for s_end, name in SENSOR_MAP.items():
            if serial_key.endswith(s_end):
                label = name
                break

        if label not in active_files:
            filename = os.path.join(OUTPUT_FOLDER, f"{label}.csv")
            f        = open(filename, 'w', newline='')
            writer   = csv.writer(f)
            # timestamp is now UTC milliseconds (wall clock)
            writer.writerow(['timestamp_utc_ms',
                             'acc_x', 'acc_y', 'acc_z',
                             'gyro_x', 'gyro_y', 'gyro_z',
                             'magn_x', 'magn_y', 'magn_z'])
            active_files[label] = {'file': f, 'writer': writer}
            print(f"[{label.upper()}] File created: {filename}")

        clean_row = row[2:]   # drop serial_key and data-type fields
        active_files[label]['writer'].writerow(clean_row)
        active_files[label]['file'].flush()


#clock sync before streaming!!!
#clock sync from Pangiotta

class MovesenseClient:
    def __init__(self, device, queue):
        self.device      = device
        self.queue       = queue
        self._ble_client = None

        
        self.utc_offset_ms: int = 0
        self._ongoing_data_update = None
        self._pending: dict[int, asyncio.Future] = {}

    #Notification handler

    async def _notification_handler(self, sender, data):
        d            = DataView(data)
        response_code = d.get_uint_8(0)
        ref_code      = d.get_uint_8(1)

        if response_code == 0x01:
            fut = self._pending.get(ref_code)
            if fut and not fut.done():
                fut.set_result(bytes(data))
            return

        #streaming data
        if response_code == 0x02: 
            self._ongoing_data_update = data
            return

        if response_code == 0x03:    
            if self._ongoing_data_update is None:
                return
            full_data = self._ongoing_data_update + data[2:]
            self._ongoing_data_update = None
            await self._parse_imu_packet(full_data)

    
    #IMU packet parser

    async def _parse_imu_packet(self, full_data):
        d           = DataView(full_data)
        sensor_tag  = self.device.name
        # Bytes 0-1: response_code + ref; bytes 2-5: sensor relative timestamp (ms)
        rel_timestamp = d.get_uint_32(2)

        for i in range(8):
            sample_rel_ts  = rel_timestamp - int((7 - i) * 1000 / 104)
            # Convert to UTC milliseconds using the per-sensor clock offset
            sample_utc_ms  = sample_rel_ts + self.utc_offset_ms

            offset = 6 + i * 3 * 4
            skip   = 3 * 8 * 4
            if offset + 2 * skip + 8 >= len(full_data):
                break

            msg_row = "{},IMU9,{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(
                sensor_tag, sample_utc_ms,
                d.get_float_32(offset),          d.get_float_32(offset + 4),      d.get_float_32(offset + 8),
                d.get_float_32(offset + skip),   d.get_float_32(offset + skip + 4), d.get_float_32(offset + skip + 8),
                d.get_float_32(offset + 2*skip), d.get_float_32(offset + 2*skip + 4), d.get_float_32(offset + 2*skip + 8),
            )
            await self.queue.put(msg_row)

    # helpers to send command and await response
    async def _send_and_wait(self, cmd_bytes: bytes, ref_code: int, timeout: float = 5.0) -> bytes:
        """Send a GATT command and wait for the matching response packet."""
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        self._pending[ref_code] = fut
        try:
            await self._ble_client.write_gatt_char(WRITE_UUID, bytearray(cmd_bytes))
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(ref_code, None)

   #now clock sync
    async def sync_clock(self):
        name = self.device.name
        print(f"[{name}] Clock sync: setting UTC time …")

        #put utc
        # Command 8, ref REF_UTCTIME, data = uint64 LE microseconds
        utc_us = int(time.time() * 1_000_000)
        cmd_put = struct.pack('<BBQ', 8, REF_UTCTIME, utc_us)
        resp    = await self._send_and_wait(cmd_put, REF_UTCTIME)

        # Check HTTP status (bytes 2-3, uint16 LE); 200 = OK
        status = struct.unpack_from('<H', resp, 2)[0]
        if status != 200:
            raise RuntimeError(f"[{name}] PUT_UTCTIME failed, HTTP status {status}")

        # get time
        # Command 4, ref REF_TIME_DETAIL, data = path as utf-8
        path     = "/Time/Detailed"
        cmd_get  = bytes([4, REF_TIME_DETAIL]) + path.encode("utf-8")
        resp     = await self._send_and_wait(cmd_get, REF_TIME_DETAIL)

        status = struct.unpack_from('<H', resp, 2)[0]
        if status != 200:
            raise RuntimeError(f"[{name}] GET /Time/Detailed failed, HTTP status {status}")

        payload      = resp[4:]
        utc_time_us  = struct.unpack_from('<q', payload, 0)[0]   # int64
        relative_ms  = struct.unpack_from('<I', payload, 8)[0]   # uint32

        # offset in milliseconds (same from Giota)
        self.utc_offset_ms = (utc_time_us // 1000) - relative_ms

        print(f"[{name}] Clock sync done. utc_offset_ms = {self.utc_offset_ms}")

    #subscribe to IMU stream
    async def _start_stream(self):
        """Send SUBSCRIBE for IMU9/104. Notifications must already be enabled."""
        cmd = bytes([1, REF_STREAM]) + "/Meas/IMU9/104".encode("utf-8")
        await self._ble_client.write_gatt_char(WRITE_UUID, bytearray(cmd))
        print(f"[{self.device.name}] Stream started.")


    async def run(self, barrier, all_clients):
        print(f"Connecting to {self.device.name}...")
        async with BleakClient(self.device.address) as client:
            self._ble_client = client
            print(f"CONNECTED: {self.device.name}. Waiting for partner...")
            await client.start_notify(NOTIFY_UUID, self._notification_handler)
            await self.sync_clock()
            await barrier.wait()
            print(f"STATUS: STREAMING")
            print(f"STARTING STREAM: {self.device.name}")
            await asyncio.gather(*[c._start_stream() for c in all_clients])

            while True:
                await asyncio.sleep(1)



async def main():
    queue = asyncio.Queue()
    print("Scanning...")
    devices        = await BleakScanner.discover(timeout=5.0)
    target_suffixes = list(SENSOR_MAP.keys())
    targets        = [
        d for d in devices
        if d.name and any(d.name.endswith(s) for s in target_suffixes)
    ]

    if len(targets) < 2:
        print(f"STATUS: ERROR - Found {len(targets)} sensor(s), need 2. "
              "Check sensors are on and in range.")
        return

    sync_barrier = asyncio.Barrier(2)

    asyncio.create_task(run_queue_consumer(queue))

    clients = [MovesenseClient(dev, queue) for dev in targets]

    tasks = [
        asyncio.create_task(client.run(sync_barrier, clients))
        for client in clients
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping …")
