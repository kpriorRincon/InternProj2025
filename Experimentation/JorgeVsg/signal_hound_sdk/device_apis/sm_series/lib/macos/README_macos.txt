Signal Hound SM200/435 Programming API for MacOSX

Email/Support: support@signalhound.com

-- Documentation -- 
https://signalhound.com/sigdownloads/SDK/online_docs/sm_api/index.html

-- Limitations --
The follow limitations apply to the ARM build of the API,
- Only USB devices are supported (B-models). Networked (C-models) are not supported.
- The API is limited to the sweep and I/Q streaming modes only.
- In sweep mode, RBW and VBW is limited to 12.5Hz.
- In the non-fast sweep mode, the sweep speed will be reduced, fast-sweep mode will be minimally impacted.
- In I/Q streaming mode, decimation is limited to values of 1, 2, and 4.
- In I/Q streaming mode, the software filter cannot be disabled. This means there can be
  spurious and aliased signals in the rejection regions. The passband will not be affected. 
  If spurious signals in the reject bands are not acceptable, we recommend
  running an additional post acquisition FIR filter.

-- Compilation Notes -- 
The API is compiled on a Macbook Pro
- M4 Pro CPU
- Sequioa 15.1
- Apple clang++ 16.0.0

-- Libusb 1.0 requirement -- 
The API depends on the libusb-1.0 USB drivers.
You will need libusb-1.0 installed in the system path. 
We recommend installing libusb-1.0 with homebrew.


