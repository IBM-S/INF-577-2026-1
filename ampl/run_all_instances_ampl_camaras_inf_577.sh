
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

#AMPL_EXECUTABLE="./../../ampl/ampl.linux-intel64/ampl"
AMPL_EXECUTABLE="./../../ampl.linux-intel64/ampl"

MODEL_FILE="mo_location_model.mod"
RUN_FILE="mo_drp_location.run"
CONFIG_FILE="conf_execution.dat" 

PROBLEM_TYPE="inf"

BENCHMARK="phub"   # cdmx | clustering | phub
INSTANCES_DIR="${BASE_DIR}/data/instances/${BENCHMARK}"
RESULTS_DIR="${BASE_DIR}/data/baselines/raw/${BENCHMARK}"

NUM_RUNS=2

declare -a INSTANCE_ORDER=(
    "wsc_0_R0p1455_pre13.dat"
    "wsc_1_R0p1268_pre17.dat"
    "wsc_2_R0p0892_pre6.dat"
    "wsc_3_R0p0628_pre7.dat"
    "wsc_4_R0p1915_pre19.dat"
    "wsc_5_R0p1708_pre5.dat"
    "wsc_6_R0p1307_pre13.dat"
    "wsc_7_R0p1438_pre15.dat"
)

# ================= INICIO DEL PROCESO =================
echo "---------------------------------------------------------"
echo " INICIANDO EXPERIMENTOS AMPL -TIPO: ${PROBLEM_TYPE}"
echo " Directorio Base: ${BASE_DIR}"
echo " Resultados en:   ${RESULTS_DIR}"
echo " Total Runs por instancia: ${NUM_RUNS}"
echo "---------------------------------------------------------"

mkdir -p ${RESULTS_DIR}

for instanceFile in ${INSTANCE_ORDER[@]}; do

    fullInstancePath="${INSTANCES_DIR}/${instanceFile}"

    if [ ! -f "${fullInstancePath}" ]; then
        echo "¡Advertencia! Archivo de instancia no encontrado: ${fullInstancePath}. Saltando..."
        continue
    fi

    instanceName=$(basename "$instanceFile" .dat)

    echo "================================================="
    echo "Procesando instancia: ${instanceFile} (${NUM_RUNS} veces)"
    echo "================================================="

    instanceResDir="${RESULTS_DIR}/${instanceName}"
    mkdir -p "${instanceResDir}"

    instanceLogFile="${instanceResDir}/execution_${instanceName}_summary.log"

    # Si existe el archivo de resumen anterior, LO BORRAMOS.
    if [ -f "${instanceLogFile}" ]; then
        # echo "   [Info] Borrando resumen anterior..."
        rm "${instanceLogFile}"
    fi

    # Creamos el archivo nuevo con el encabezado CSV
    echo "Run,Time_s" > "${instanceLogFile}"

    for (( run=1; run <=${NUM_RUNS}; run++)); do
        echo "  Ejecución ${run}/${NUM_RUNS} para la instancia ${instanceFile}..."
        startTime="$(date +%s.%N)"

        outputDir="${instanceResDir}/run_${run}"

        if [ -d "${outputDir}" ]; then
            rm -rf "${outputDir}"
        fi
        mkdir -p "${outputDir}"

        fullLogFile="${outputDir}/ampl_log_full.txt"
        paretoFile="${outputDir}/pareto_front.txt"

        ${AMPL_EXECUTABLE} <<EOF > "${fullLogFile}"
        reset;
        model ${MODEL_FILE};
        data "${fullInstancePath}";
        data "${CONFIG_FILE}";
        
        include "${RUN_FILE}";
EOF

        endTime="$(date +%s.%N)"
        duration=$(echo "$endTime - $startTime" | bc)

        paretoFile="$outputDir/pareto_front.txt"
        grep -A 999 "F1(Coverage) F2(Cost)" "${fullLogFile}" | tail -n +3 > "${paretoFile}"

        echo "${run},${duration}" >> ${instanceLogFile}
        echo "Completado en ${duration}s. Resultados guardados en ${outputDir}"

    done
    echo "Instancia ${instanceFile} procesada."
done

echo "--- Todas las instancias han sido procesadas. ---"