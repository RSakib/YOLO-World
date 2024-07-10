<#
    Activate Python Virtual Environment for Development
    From the same directory of this file {activate.ps1}, run the following command
    ./activate
#>

# Set virtual environment folder path
$devPath = '.venv'

# Output to console
Write-Host "Checking for python environment...";

# Check if virtual environment folder exists
if(![System.IO.Directory]::Exists($devPath)){
    # Create python virtual environment if .venv folder does not exist
    Write-Host "Creating python environment...";

    # Command to create .venv
    python -m venv .\.venv
}

# Output to console
# Write-Host "Setting environment vars...";
# $env:PROJECT_NAME="MUGIC";

# Output to console
Write-Host "Initializing process..."

# Command to activate virtual environment
.\.venv\Scripts\Activate

# Command to install the specific version of pip for virtual environment
python -m pip install --upgrade pip==24.1.2

# Command to install dependencies specified in {requirements.txt}
Write-Host "Downloading dependencies..."
python -m pip install --no-cache-dir -r requirements\basic_requirements.txt
python -m pip install --no-cache-dir -r requirements\demo_requirements.txt
python -m pip install --no-cache-dir -r requirements\onnx_requirements.txt