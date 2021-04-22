input = "/home/koosk/data/tmp/unet/";
output = "/home/koosk/data/tmp/unet-png/"


list = getFileList(input);
for (i = 0; i < list.length; i++){
	open(input + list[i]);
    run("Grays");
    saveAs("PNG", output + list[i]);
	close();
}