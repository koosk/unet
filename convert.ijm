input = "/home/koosk/data/images/renard/unet/HCT116_Intro_Replicate1/train/label/";


list = getFileList(input);
for (i = 0; i < list.length; i++){
	open(input + list[i]);
    run("Multiply...", "value=255.000");
	setOption("ScaleConversions", true);
	run("8-bit");
	run("Multiply...", "value=255.000");
	run("Save");
	close();
}