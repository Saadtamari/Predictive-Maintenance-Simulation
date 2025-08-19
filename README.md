# DA380 Predictive Maintenance Simulation

**Course:** Data Modeling and Simulation (DA380)  
**Project:** Predictive Maintenance Simulation for Smart Manufacturing  
**Implementation:** Python + Streamlit (Excel Alternative)  
**Semester:** Summer 2025

## 🎯 Project Overview

This project simulates predictive maintenance strategies for a smart manufacturing environment, comparing Reactive, Preventive, and Predictive maintenance approaches across multiple machine types (CNC machines, Conveyor belts, Robotic arms).

## 📋 Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run predictive_maintenance_sim.py
```

### 3. Open Browser
The application will automatically open at: `http://localhost:8501`

## 📊 How to Use

1. **Configure Machines**: Use the sidebar to adjust machine parameters
2. **Set Simulation Period**: Choose simulation duration (30-365 days)
3. **Run Scenarios**: Click "Run All Scenarios" to execute simulation
4. **Analyze Results**: Review comparative charts and detailed analysis
5. **Export Data**: Use the export functionality for report generation

## 📁 File Structure

```
DA380_Final_Project/
├── predictive_maintenance_sim.py    # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── Final_Report.pdf                # Project report
└── screenshots/                    # Application screenshots
    ├── main_interface.png
    ├── simulation_results.png
    └── comparative_analysis.png
```

## 🔧 Features

- **Interactive Parameter Configuration**
- **3 Maintenance Strategies Comparison**
- **Real-time Visualization**
- **Machine-level Performance Analysis**
- **Cost-Benefit Analysis**
- **Export Capabilities**

## 📈 Simulation Parameters

### Default Machine Configuration:
- **CNC Machines**: 5 units, 2.5 failures/1000hrs, $1500 maintenance cost
- **Conveyor Belts**: 8 units, 1.8 failures/1000hrs, $800 maintenance cost  
- **Robotic Arms**: 3 units, 3.2 failures/1000hrs, $2500 maintenance cost

### Maintenance Strategies:
- **Reactive**: Fix after failure
- **Preventive**: Scheduled maintenance at 80% of expected failure time
- **Predictive**: Optimized scheduling at 85% of expected failure time

## 🎓 Academic Compliance

This implementation fully meets DA380 project requirements:
- ✅ Multiple machine types (3+)
- ✅ Failure rate simulation
- ✅ Maintenance cost analysis
- ✅ Multiple scenario comparison (3 strategies)
- ✅ Statistical analysis and visualization
- ✅ Comprehensive reporting capabilities

## 💡 Key Insights Generated

The simulation provides:
- **Cost-effectiveness ranking** of maintenance strategies
- **Machine availability optimization** recommendations
- **Resource allocation** insights
- **ROI analysis** for maintenance investments

## 🔍 Troubleshooting

### Common Issues:
1. **Port already in use**: Try `streamlit run predictive_maintenance_sim.py --server.port 8502`
2. **Package conflicts**: Create a virtual environment: `python -m venv venv`
3. **Performance issues**: Reduce simulation period or machine count for faster execution

### System Requirements:
- **RAM**: Minimum 4GB (8GB recommended)
- **Browser**: Chrome, Firefox, or Safari (latest versions)
- **Internet**: Required for initial package installation

## 📞 Support

For technical issues or questions:
- Check the troubleshooting section above
- Review Streamlit documentation: https://docs.streamlit.io/
- Contact course instructor for project-specific guidance

## 📄 License

Academic project for DA380 course - Summer 2025
