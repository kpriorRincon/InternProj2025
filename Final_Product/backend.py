import sockshandler

import async

async def backend_transmit(message):
    #run the bash script through SSH
    ssh_host = 'empire@empire'
    cmd = f' ./transmit.sh "{message}"'


