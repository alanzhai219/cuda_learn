echo "logical CPU number:"
data=`cat /proc/cpuinfo | grep "processor" | wc -l`
echo $data
echo "..................................................................................."
#
echo "physical CPU number:"
data=`cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l`
echo $data
echo "..................................................................................."
#
echo "core number in a physical CPU:"
data=`cat /proc/cpuinfo | grep "cpu cores" | uniq | awk -F: '{print $2}'`
echo $data
echo "..................................................................................."
# 
echo "core index:"
data=`cat /proc/cpuinfo | grep "core id" | sort | uniq `
echo $data
echo "..................................................................................."
# 
echo "Inst support:"
data=`cat /proc/cpuinfo | grep flags | grep ht`
echo $data
echo "..................................................................................."
#
data=`cat /proc/cpuinfo | grep "siblings"`
echo $data

