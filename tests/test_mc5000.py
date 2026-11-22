from unittest import mock

import pytest
from bleak import BLEDevice

from skyrc_ble import Mc5000
from skyrc_ble.mc5000 import build_frame, build_program_payload
from skyrc_ble.models import (
    Mc5000BatteryType,
    Mc5000ChannelData,
    Mc5000ChannelStatus,
    Mc5000Error,
    Mc5000Operation,
    Mc5000Program,
)

NIMH_CHARGE_300MA = Mc5000Program(
    battery_type=Mc5000BatteryType.NIMH,
    operation=Mc5000Operation.CHARGE,
    capacity=2000,
    charge_current=0.3,
    discharge_current=0.5,
    charge_voltage=1650,
    discharge_voltage=900,
)


@pytest.fixture(autouse=True)
def _fast_delays():
    with mock.patch("skyrc_ble.mc5000.CONNECT_SETTLE_DELAY", 0), mock.patch(
        "skyrc_ble.mc5000.INTER_COMMAND_DELAY", 0
    ):
        yield


async def _connected(name="#Charger-4036"):
    mc5000 = Mc5000(BLEDevice("00:01:02:03:04:05", name, None))
    await mc5000.connect()
    return mc5000


def test_build_frame():
    assert build_frame(0x91, [0x01]) == bytes.fromhex("0f 03 91 01 92")
    assert build_frame(0xFE, [0x00]) == bytes.fromhex("0f 03 fe 00 fe")


@pytest.mark.asyncio
async def test_mc5000_state(mock_mc5000_bleak):
    mc5000 = await _connected()
    await mc5000.update()

    assert mc5000.model == "MC5000"
    assert mc5000.state.channels[0] == Mc5000ChannelData(
        type=Mc5000BatteryType.NIMH,
        operation=Mc5000Operation.CHARGE,
        status=Mc5000ChannelStatus.STANDBY,
        error=Mc5000Error.NONE,
        time=0,
        voltage=1.295,
        current=0.0,
        capacity=0,
        temperature=0,
        resistance=0,
    )
    assert mc5000.state.channels[2].type == Mc5000BatteryType.LIION
    assert all(channel is not None for channel in mc5000.state.channels)


@pytest.mark.asyncio
async def test_mc5000_reassembles_split_notifications(mock_mc5000_bleak):
    mock_mc5000_bleak.split_responses = True
    mc5000 = await _connected()
    await mc5000.update()
    assert mc5000.state.channels[0].voltage == 1.295


@pytest.mark.asyncio
async def test_mc5000_start_nimh_charge_matches_hardware_capture(mock_mc5000_bleak):
    mc5000 = await _connected()
    await mc5000.start_program(0, NIMH_CHARGE_300MA)

    # Frame sent to a real MC5000, which acknowledged it and charged at 300 mA.
    assert mc5000._client.sent_data[-1] == bytes.fromhex(
        "0f 2a 94 01 00 01 2c 01 f4 07 d0 06 72 03 84 00 64 00 64 00 0a 00 0a 01 00 06 "
        "00 00 00 3c 00 00 01 2c 00 03 0c e4 00 00 00 00 00 cc"
    )


@pytest.mark.asyncio
async def test_mc5000_start_nimh_discharge_matches_hardware_capture(mock_mc5000_bleak):
    mc5000 = await _connected()
    program = Mc5000Program(
        battery_type=Mc5000BatteryType.NIMH,
        operation=Mc5000Operation.DISCHARGE,
        capacity=3000,
        charge_current=0.3,
        discharge_current=0.5,
        charge_voltage=1650,
        discharge_voltage=1000,
    )
    await mc5000.start_program(1, program)

    assert mc5000._client.sent_data[-1] == bytes.fromhex(
        "0f 2a 94 02 03 01 2c 01 f4 0b b8 06 72 03 e8 00 64 00 64 00 0a 00 0a 01 00 06 "
        "00 00 00 3c 00 00 01 2c 00 03 0c e4 00 00 00 00 00 20"
    )


@pytest.mark.asyncio
async def test_mc5000_stop(mock_mc5000_bleak):
    mc5000 = await _connected()
    await mc5000.stop_charge(0)
    assert mc5000._client.sent_data[-1] == bytes.fromhex("0f 03 fe 01 ff")
    await mc5000.stop_charge_multi(0)
    assert mc5000._client.sent_data[-1] == bytes.fromhex("0f 03 fe 00 fe")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"charge_voltage": 4300, "battery_type": Mc5000BatteryType.LIION}, "1..4200"),
        ({"operation": Mc5000Operation.STORAGE}, "not available for NIMH"),
        ({"charge_current": 3.5}, "between 0 and 3.0 A"),
        ({"discharge_current": 2.5}, "between 0 and 2.0 A"),
        ({"charge_current": 0.3005}, "0.001 A increments"),
        ({"trickle_current": 15}, "10 mA increments"),
    ],
)
def test_mc5000_program_validation(changes, message):
    fields = {**NIMH_CHARGE_300MA.__dict__, **changes}
    with pytest.raises(ValueError, match=message):
        build_program_payload(0x01, Mc5000Program(**fields))


@pytest.mark.asyncio
async def test_mc5000_rejects_invalid_channels(mock_mc5000_bleak):
    mc5000 = await _connected()
    with pytest.raises(ValueError):
        await mc5000.start_program(4, NIMH_CHARGE_300MA)
    with pytest.raises(ValueError):
        await mc5000.stop_charge_multi(0x10)


@pytest.mark.asyncio
async def test_mc5000_request_after_disconnect_raises_bleak_error(mock_mc5000_bleak):
    from bleak.exc import BleakError

    mc5000 = await _connected()
    mc5000._client = None  # what the disconnected callback does
    with pytest.raises(BleakError):
        await mc5000.stop_charge(0)


# Frames captured from a real MC5000 (firmware 2.02) on 2026-11-22.
SETTINGS_FRAME = bytes.fromhex("0f0996080001000101 00a1".replace(" ", ""))
SETTINGS_AFTER_WRITE = bytes.fromhex("0f09960800010003010 0a3".replace(" ", ""))
SET_ACK = bytes.fromhex("0f049510 01a6".replace(" ", ""))
DEVICE_INFO_FRAME = bytes.fromhex("0f1357003130303231340400000000020202 00ce57".replace(" ", ""))
USB_IDLE = bytes.fromhex("0f109b0000000000000000000000000000 9b".replace(" ", ""))
USB_PHONE = bytes.fromhex("0f109b0002087a3ba0000000e60000002f0f")


def test_parse_settings():
    from skyrc_ble.mc5000 import frame_checksum_ok, parse_settings

    assert frame_checksum_ok(SETTINGS_FRAME)
    s = parse_settings(SETTINGS_FRAME)
    assert (s.keypress_beep, s.notify_beep, s.completion_repeat) == (0, 1, False)
    assert (s.backlight, s.warning, s.screen_timeout) == (1, 1, 0)
    assert parse_settings(SETTINGS_AFTER_WRITE).backlight == 3


def test_build_settings_payload_matches_hardware_test():
    from dataclasses import replace

    from skyrc_ble.mc5000 import build_frame, build_settings_payload, parse_settings

    s = replace(parse_settings(SETTINGS_FRAME), backlight=3)
    payload = build_settings_payload(s)
    # The exact payload that changed the backlight on the real charger.
    assert payload.hex(" ") == "00 00 01 00 03 01 00 00 00 00 00 00 00 00 00 00 00"
    assert build_frame(0x95, payload)[:3] == bytes([0x0F, 19, 0x95])
    with pytest.raises(ValueError):
        build_settings_payload(replace(s, backlight=0))
    with pytest.raises(ValueError):
        build_settings_payload(replace(s, screen_timeout=7))
    with pytest.raises(ValueError):
        build_settings_payload(replace(s, keypress_beep=4))


def test_parse_usb_status():
    from skyrc_ble.mc5000 import frame_checksum_ok, parse_usb_status

    assert frame_checksum_ok(USB_PHONE)
    u = parse_usb_status(USB_PHONE)
    assert u.status == 2
    assert u.current == 2.17 and u.voltage == 15.264
    assert u.energy == 0.23 and u.runtime == 47
    assert u.power == 33.12
    idle = parse_usb_status(USB_IDLE)
    assert (idle.status, idle.power, idle.runtime) == (0, 0, 0)


def test_device_info():
    from skyrc_ble.mc5000 import device_info_payload, frame_checksum_ok, parse_device_info

    assert frame_checksum_ok(DEVICE_INFO_FRAME)
    info = parse_device_info(DEVICE_INFO_FRAME)
    assert info.model_code == "100214"
    assert info.firmware == "2.02" and info.hardware == "2"
    assert device_info_payload("DC:23:53:64:36:40").hex(" ") == (
        "00 40 36 64 53 23 dc 00 00 00 00 00 00 00 00 00 00"
    )
    with pytest.raises(ValueError):
        device_info_payload("1A2B3C4D-0000-0000-0000-000000000000")
